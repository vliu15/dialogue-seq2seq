''' This file contains all functions used to calculate evaluation metrics '''
import torch
import torch.nn.functional as F

import seq2seq.Constants as Constants


def cal_performance(pred, gold, smoothing=False, mmi_factor=1.0):
    '''
    Calculate accuracy and loss with
    1) label smoothing if specified
    2) maximal mutual information (MMI) if specified
    3) perplexity
    '''
    if mmi_factor > 0:
        #- Calculate CE loss with MMI objective
        pred_session, pred_no_session = torch.split(pred, int(pred.shape[0]/2), dim=0)
        pred_no_session = pred_no_session.detach()
        loss = cal_mmi_loss(pred_session, pred_no_session, gold, smoothing=smoothing, mmi_factor=mmi_factor)
        with torch.no_grad():
            nll = cal_mle_loss(pred_session, gold, smoothing)
        pred = (pred_session - pred_no_session).max(1)[1]
    else:
        #- Calculate CE loss with MLE objective
        loss = cal_mle_loss(pred, gold, smoothing)
        pred = pred.max(1)[1]
        nll = loss
    
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct, nll
    
def cal_mmi_loss(pred_session, pred_no_session, gold, smoothing=True, mmi_factor=1.0):
    '''
    Calculate MMI objective, apply label smoothing if needed.

    MMI objective:
        r* = argmax_r {log P(r|r_) - lamb * log P(r)}
    where r is the session-infused response,
          r_ is the session-dry response,
          lamb is the weighting factor (lamb=0.0 is MLE)
    '''
    gold = gold.contiguous().view(-1)
    one_hot = torch.zeros_like(pred_session).scatter(1, gold.view(-1, 1), 1)

    if smoothing:
        eps = 0.1
        n_class = pred_session.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

    ses_output_sftmx = F.log_softmax(pred_session, dim=1)
    no_ses_outout_sftmx = F.log_softmax(pred_no_session, dim=1)
    final_sftmax = ses_output_sftmx - mmi_factor * no_ses_outout_sftmx

    non_pad_mask = gold.ne(Constants.PAD)
    loss = -(one_hot * final_sftmax).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).sum()

    return loss

def cal_mle_loss(pred, gold, smoothing):
    '''
    Calculate cross entropy loss, apply label smoothing if needed.
    
    MLE objective:
        r* = argmax_r {log P(r|r_)}
    where r is the session-infused response,
          r_ is the session-dry response
    '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss
