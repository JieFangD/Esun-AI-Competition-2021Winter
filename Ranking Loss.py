def approx_ranks(logits):
    r"""Computes approximate ranks given a list of logits.
    Given a list of logits, the rank of an item in the list is one plus the total
    number of items with a larger logit. In other words,
    rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},
    where "I" is the indicator function. The indicator function can be
    approximated by a generalized sigmoid:
    I_{s_j < s_i} \approx 1/(1 + exp(-(s_j - s_i)/temperature)).
    This function approximates the rank of an item using this sigmoid
    approximation to the indicator function. This technique is at the core
    of "A general approximation framework for direct optimization of
    information retrieval measures" by Qin et al.
    Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    Returns:
    A `Tensor` of ranks with the same shape as logits.
    """
    list_size = logits.size()[1]
    x = torch.tile(torch.unsqueeze(logits, 2), (1, 1, list_size))
    y = torch.tile(torch.unsqueeze(logits, 1), (1, list_size, 1))
    pairs = torch.sigmoid(y - x)
    return torch.sum(pairs, axis=-1) + .5

def ndcg(labels, ranks=None, perm_mat=None):
    """Computes NDCG from labels and ranks.
    Args:
      labels: A `Tensor` with shape [batch_size, list_size], representing graded
        relevance.
      ranks: A `Tensor` of the same shape as labels, or [1, list_size], or None.
        If ranks=None, we assume the labels are sorted in their rank.
      perm_mat: A `Tensor` with shape [batch_size, list_size, list_size] or None.
        Permutation matrices with rows correpond to the ranks and columns
        correspond to the indices. An argmax over each row gives the index of the
        element at the corresponding rank.
    Returns:
      A `tensor` of NDCG, ApproxNDCG, or ExpectedNDCG of shape [batch_size, 1].
    """
    if ranks is not None and perm_mat is not None:
        raise ValueError('Cannot use both ranks and perm_mat simultaneously.')

    if ranks is None:
        list_size = labels.size()[1]
        ranks = (torch.arange(list_size) + 1).cuda()
    discounts = 1. / torch.log1p(ranks)
    gains = 2**labels - 1
    if perm_mat is not None:
        gains = torch.sum(perm_mat * torch.squeeze(gains, 1), axis=-1)
    dcg = torch.sum(gains * discounts, axis=-1, keepdim=True)
    normalized_dcg = dcg * inverse_max_dcg(labels)

    return normalized_dcg

def inverse_max_dcg(labels,
                    gain_fn=lambda labels: 2**labels - 1,
                    rank_discount_fn=lambda rank: 1. / torch.log1p(rank),
                    topn=None):
    """Computes the inverse of max DCG.
    Args:
      labels: A `Tensor` with shape [batch_size, list_size]. Each value is the
        graded relevance of the corresponding item.
      gain_fn: A gain function. By default this is set to: 2^label - 1.
      rank_discount_fn: A discount function. By default this is set to:
        1/log(1+rank).
      topn: An integer as the cutoff of examples in the sorted list.
    Returns:
      A `Tensor` with shape [batch_size, 1].
    """
    if topn is None:
        topn = labels.size()[1]
    ideal_sorted_labels = torch.topk(labels, k=topn)[0]
    rank = (torch.arange(ideal_sorted_labels.size()[1]) + 1).cuda()
    discounted_gain = gain_fn(ideal_sorted_labels) * rank_discount_fn(rank)
    discounted_gain = torch.sum(discounted_gain, axis=1, keepdim=True)
    return torch.where(discounted_gain > 0, 1/discounted_gain, torch.zeros_like(discounted_gain))

def ApproxNDCGLoss(output, y, temp=0.1):
    batch, t, dim = output.shape
    output = output.reshape(-1,dim)/temp
    y = y.reshape(-1,dim)
    ranks = approx_ranks(output)
    return torch.mean(-ndcg(y, ranks))

class GumbelSampler(object):
    """Random sampler for sampling gumbel distributed logits."""

    def __init__(self, name=None, sample_size=8, temperature=1.0, seed=None):
        """Constructor."""
        self._name = name
        self._sample_size = sample_size
        self._temperature = temperature
        self._seed = seed

    def sample_gumbel(self, shape, eps=1e-20, seed=None):
        if seed:
            torch.manual_seed(seed)
        u = torch.rand(shape).cuda()
        return -torch.log(-torch.log(u + eps) + eps)
        
    def sample(self, labels, logits, weights=None):
        """Samples scores from Concrete(logits).
        Args:
          labels: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size]
            same as `logits`, representing graded relevance. Or in the diversity
            tasks, a `Tensor` (or `RaggedTensor`) with shape [batch_size, list_size,
            subtopic_size]. Each value represents relevance to a subtopic, 1 for
            relevent subtopic, 0 for irrelevant, and -1 for paddings. When the
            actual subtopic number of a query is smaller than the `subtopic_size`,
            `labels` will be padded to `subtopic_size` with -1.
          logits: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size].
            Each value is the ranking score of the corresponding item.
          weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
            weights, or a `Tensor` or `RaggedTensor` with shape [batch_size,
            list_size] for item-wise weights. If None, the weight of a list in the
            mini-batch is set to the sum of the labels of the items in that list.
        Returns:
          A tuple of expanded labels, logits, and weights where the first dimension
          is now batch_size * sample_size. Logit Tensors are sampled from
          Concrete(logits) while labels and weights are simply tiled so the
          resulting
          Tensor has the updated dimensions.
        """
        batch_size = labels.size()[0]
        list_size = labels.size()[1]

        # Expand labels.
        expanded_labels = torch.unsqueeze(labels, 1)
        expanded_labels = torch.tile(expanded_labels, (1,self._sample_size,1))
        expanded_labels = torch.reshape(expanded_labels, (batch_size*self._sample_size, list_size))

        # Sample logits from Concrete(logits).
        sampled_logits = torch.unsqueeze(logits, 1)
        sampled_logits = torch.tile(sampled_logits, (1,self._sample_size,1))
        sampled_logits += self.sample_gumbel(
            [batch_size, self._sample_size, list_size], seed=self._seed)
        sampled_logits = torch.reshape(sampled_logits, (batch_size*self._sample_size, list_size))

        is_label_valid = (expanded_labels >= 0)
        if is_label_valid.dim() > 2:
            is_label_valid = torch.any(is_label_valid, dim=-1)
        sampled_logits = torch.where(
          is_label_valid, sampled_logits / self._temperature,
          np.log(1e-20) * torch.ones_like(sampled_logits))
        sampled_logits = torch.log(torch.nn.functional.softmax(sampled_logits,1) + 1e-20)

        expanded_weights = weights
        if expanded_weights is not None:
            true_fn = lambda: torch.unsqueeze(torch.unsqueeze(expanded_weights, 1), 1)
            false_fn = lambda: torch.unsqueeze(expanded_weights, 1)
            expanded_weights = tf.cond(
                pred=torch.equal(torch.dim(expanded_weights), 1),
                true_fn=true_fn,
                false_fn=false_fn)
            expanded_weights = torch.tile(expanded_weights, (1,self._sample_size,1))
            expanded_weights = torch.reshape(expanded_weights, (batch_size*self._sample_size, -1))

        return expanded_labels, sampled_logits, expanded_weights
    
def GumbelApproxNDCGLoss(output, y, temp=0.1, sample_size=8, gumbel_temp=1.0):
    gumbel = GumbelSampler(sample_size=sample_size, temperature=gumbel_temp)
    batch, t, dim = output.shape
    output = output.reshape(-1,dim)
    y = y.reshape(-1,dim)
    gbl_labels, gbl_logits, gbl_weights = gumbel.sample(y, output)
    return ApproxNDCGLoss(torch.unsqueeze(gbl_logits,1), torch.unsqueeze(gbl_labels,1), temp=0.1)

def _apply_pairwise_op(op, tensor):
    """Applies the op on tensor in the pairwise manner."""
    return op(torch.unsqueeze(tensor, 2), torch.unsqueeze(tensor, 1))

def _pairwise_comparison(labels, logits, mask):
    r"""Returns pairwise comparison `Tensor`s.
    Given a list of n items, the labels of graded relevance l_i and the logits
    s_i, we form n^2 pairs. For each pair, we have the following:
                        /
                        | 1   if l_i > l_j for valid l_i and l_j.
    * `pairwise_labels` = |
                        | 0   otherwise
                        \
    * `pairwise_logits` = pairwise_logits_op(s_i, s_j)
    Args:
      labels: A `Tensor` with shape [batch_size, list_size].
      logits: A `Tensor` with shape [batch_size, list_size].
      mask: A `Tensor` with shape [batch_size, list_size] indicating which entries
        are valid for computing the pairwise comparisons.
    Returns:
      A tuple of (pairwise_labels, pairwise_logits) with each having the shape
      [batch_size, list_size, list_size].
    """
    # Compute the difference for all pairs in a list. The output is a Tensor with
    # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
    # the information for pair (i, j).
    pairwise_label_diff = _apply_pairwise_op(torch.subtract, labels)
    pairwise_logits = _apply_pairwise_op(torch.subtract, logits)
    # Only keep the case when l_i > l_j.
    pairwise_labels = (pairwise_label_diff > 0)
    valid_pair = _apply_pairwise_op(torch.logical_and, mask)
    pairwise_labels = pairwise_labels*valid_pair
    return pairwise_labels, pairwise_logits

def UniqueSoftmaxLoss(output, y, temp=1):
    batch, t, dim = output.shape
    logits = output.reshape(-1,dim)
    labels = y.reshape(-1,dim)
    mask = (labels >= 0)
    pairwise_labels, _ = _pairwise_comparison(labels, logits, mask)
    # Used in denominator to compute unique softmax probability for each doc.
    denominator_logits = torch.unsqueeze(logits, axis=1)*pairwise_labels
    denominator_logits = torch.concat(
        (denominator_logits, torch.unsqueeze(logits, axis=2)), axis=2)
    denominator_mask = torch.concat(
        (pairwise_labels, torch.unsqueeze(torch.ones_like(logits), axis=2)), axis=2)
    denominator_logits = torch.where(
        denominator_mask > 0.0, denominator_logits, -1e-3 +
        torch.min(denominator_logits)*torch.ones_like(denominator_logits))
    logits_max = torch.max(denominator_logits, dim=-1, keepdims=True)[0]
    # Subtract the max so that exp(denominator_logits) is numerically valid.
    denominator_logits -= logits_max
    _logits = logits - torch.squeeze(logits_max, axis=-1)
    # Set gains for loss weights.
    gains = 2**labels - 1
    # Compute the softmax loss for each doc.
    per_doc_softmax = -_logits + torch.log(
        torch.sum(torch.exp(denominator_logits) * denominator_mask, dim=-1))
    losses = torch.sum(per_doc_softmax * gains, dim=1, keepdims=True)
    return torch.mean(losses)

def SoftmaxLoss(output, y, temp=1):
    batch, t, dim = output.shape
    output = output.reshape(-1,dim)/temp
    nonzero = torch.sum(y, 1) > 0
    y = y.reshape(-1,dim)
    y = y/(torch.sum(y, dim=1, keepdim=True) + 1e-20)
    return torch.mean(-1*torch.sum(y*torch.log(torch.nn.functional.softmax(output,1) + 1e-20), dim=1))