"""Development-only publication guard; never reads future evaluation labels."""
def eligible(candidate, previous, initial):
    return (candidate['old']['mean_client_auc'] >= initial['old']['mean_client_auc']-.03
            and all(candidate[d]['fpr'] <= previous[d]['fpr']+.02
                    and candidate[d]['tpr'] >= previous[d]['tpr']-.02
                    for d in ['old','new']))
