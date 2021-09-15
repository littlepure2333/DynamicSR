import torch


def spatial_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)  # (b,c,h,w)->(b,c,hw)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)  # (b,c,hw)
    s = norm_fm.transpose(1, 2).bmm(norm_fm)  # (b,c,hw)->(b,hw,c)*(b,c,hw)->(b,hw,hw)
    s = s.unsqueeze(1) # (b,hw,hw)->(b,1,hw,hw)
    return s


def channel_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    s = norm_fm.bmm(norm_fm.transpose(1, 2))  # (b,c,hw)*(b,hw,c)->(b,c,c)
    s = s.unsqueeze(1) # (b,c,c)->(b,1,c,c)
    return s


def batch_similarity(fm):
    """batch similarity"""
    fm = fm.view(fm.size(0), -1)  # (b,chw)
    Q = torch.mm(fm, fm.transpose(0, 1)) # (b,chw)*(chw,b)->(b,b)
    normalized_Q = Q / torch.norm(Q, 2, dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q


def FSP(fm1, fm2):
    # (b,c,h,w), (b,c,h,w)
    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)  # (b,c,hw)
    fm2 = fm2.view(fm2.size(0), -1, fm2.size(1))
    res = torch.bmm(fm1, fm2)
    return res.unsqueeze(1)


def AT(feature_map):
    fm = torch.sum(torch.pow(feature_map, 2), 1).unsqueeze(1)
    fm = fm / torch.sum(fm, [2, 3]).unsqueeze(2).unsqueeze(3)
    return fm


if __name__ == '__main__':
    print(__file__)
