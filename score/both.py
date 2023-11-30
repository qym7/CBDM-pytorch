import numpy as np
import torch
import types
from tqdm import tqdm, trange
import os

from .inception import InceptionV3
from .fid import calculate_frechet_distance, torch_cov

from .improved_prd import IPR
from .prd_score import compute_prd_from_embedding, prd_to_max_f_beta_pair


device = torch.device('cuda:0')


def get_inception_and_fid_score(images, labels, fid_cache, num_images=None,
                                splits=10, batch_size=50,
                                use_torch=False,
                                verbose=False,
                                parallel=False, prd=False, FLAGS=None):
    """when `images` is a python generator, `num_images` should be given"""

    print('start calculation inception features')
    if num_images is None and isinstance(images, types.GeneratorType):
        raise ValueError(
            "when `images` is a python generator, "
            "`num_images` should be given")

    # if num_images is None:
    num_images = len(images)
    print(fid_cache)

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    if use_torch:
        fid_acts = torch.empty((num_images, 2048)).to(device)
        is_probs = torch.empty((num_images, 1008)).to(device)
    else:
        fid_acts = np.empty((num_images, 2048))
        is_probs = np.empty((num_images, 1008))

    iterator = iter(tqdm(
        images, total=num_images,
        dynamic_ncols=True, leave=False, disable=not verbose,
        desc="get_inception_and_fid_score"))

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            if use_torch:
                fid_acts[start: end] = pred[0].view(-1, 2048)
                is_probs[start: end] = pred[1]
            else:
                fid_acts[start: end] = pred[0].view(-1, 2048).cpu().numpy()
                is_probs[start: end] = pred[1].cpu().numpy()
        start = end
    print('end calculation inception features')

    # Inception Score
    print('calculate inception score')
    scores = []
    if not isinstance(fid_acts, np.ndarray):
        fid_acts = fid_acts.numpy()
    for i in range(splits):
        part = is_probs[
            (i * is_probs.shape[0] // splits):
            ((i + 1) * is_probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        is_score = (torch.mean(scores).cpu().item(),
                    torch.std(scores).cpu().item())
    else:
        is_score = (np.mean(scores), np.std(scores))

    # FID Score
    print('calculate fid')
    f = np.load(fid_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        s1 = torch_cov(fid_acts, rowvar=False)
        m2 = torch.tensor(m2).to(m1.dtype).to(device)
        s2 = torch.tensor(s2).to(s1.dtype).to(device)
    else:
        m1 = np.mean(fid_acts, axis=0)
        s1 = np.cov(fid_acts, rowvar=False)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2, use_torch=use_torch)

    # prd
    print('calculate prd (F_beta)')
    prd_score = (0, 0)
    if FLAGS.prd and len(fid_acts)==50000:
        # import pdb; pdb.set_trace()
      
        print(FLAGS.data_type)
        if FLAGS.data_type == "cifar100" or FLAGS.data_type == "cifar100lt":
           feats = np.load('/mnt/workspace/dlly/ucm3/stats/cifar100_feats.npy')
        elif FLAGS.data_type == "cifar10" or FLAGS.data_type == "cifar10lt":
           feats = np.load('/mnt/workspace/dlly/ucm3/stats/cifar10_feats.npy')
        feats = torch.Tensor(feats)
        if isinstance(fid_acts, np.ndarray):
            fid_acts = torch.Tensor(fid_acts)
        num_clusters = len(np.unique(labels)) * 20
        prd_data = compute_prd_from_embedding(
            eval_data=fid_acts,
            ref_data=feats,
            num_clusters=num_clusters,
            num_angles=1001,
            num_runs=10,
            enforce_balance=True)
        prd_data = compute_prd_from_embedding(eval_data=fid_acts,ref_data=feats,num_clusters=num_clusters,num_angles=1001,num_runs=10,enforce_balance=True)
        prd_score = prd_to_max_f_beta_pair(prd_data[0], prd_data[1], beta=8) # precision/recall
        # print('prd_score', prd_score)

    # improved prd
    print('calculate improved prd (precision/recall)')
    im_prd = (0, 0)
    if FLAGS.improved_prd and len(fid_acts)==50000:
        print(FLAGS.data_type)
        if FLAGS.data_type == "cifar100" or FLAGS.data_type == "cifar100lt":
           feats = np.load('/mnt/workspace/dlly/ucm3/stats/cifar100_feats.npy')
        elif FLAGS.data_type == "cifar10" or FLAGS.data_type == "cifar10lt":
           feats = np.load('/mnt/workspace/dlly/ucm3/stats/cifar10_feats.npy')
        if isinstance(fid_acts, torch.Tensor):
            fid_acts = fid_acts.numpy()
        ipr = IPR(32, k=5, num_samples=50000, model='InceptionV3')
        ipr.compute_manifold_ref(None, feats=feats)  # args.path_real can be either directory or pre-computed manifold file
        metric = ipr.precision_and_recall(images, subject_feats=fid_acts)
        im_prd = (metric.precision, metric.recall)
        # print('precision =', metric.precision)
        # print('recall =', metric.recall)

    print('fid', fid_score)
    print('is', is_score)
    print('prd_score', prd_score)
    print('Improved precison', im_prd[0], 'Improved recall', im_prd[1])

    return is_score, fid_score, prd_score, im_prd
