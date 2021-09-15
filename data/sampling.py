import numpy as np
import random
import matplotlib.pyplot as plt
import pyexr
from scipy import ndimage


########################################################
# def get_variance_map(data, patch_size, relative=False):
#     data = data[:, :, np.newaxis] if data.ndim < 3 else data
#     mean = ndimage.uniform_filter(data, size=(patch_size, patch_size, 1))
#     sqrmean = ndimage.uniform_filter(data ** 2, size=(patch_size, patch_size, 1))
#     variance = np.maximum(sqrmean - mean ** 2, 0)
#     variance = variance / np.maximum(mean ** 2, 1e-2) if relative else variance
#     variance = variance.max(axis=2)
#     variance = np.minimum(variance ** (1.0 / 2.2), 1.0)
#     return variance / variance.max()
#
#
# def get_importance_map(buffers, metrics, weights, patch_size):
#     if len(metrics) != len(buffers):
#         metrics = [metrics[0]] * len(buffers)
#
#     if len(weights) != len(buffers):
#         weights = [weights[0]] * len(buffers)
#
#     importance_map = None
#     for buf, metric, weight in zip(buffers, metrics, weights):
#         if metric == "uniform":
#             cur = np.ones(buf.shape[:2], dtype=np.float)
#         elif metric == "variance":
#             cur = get_variance_map(buf, patch_size, relative=False)
#         elif metric == "relvar":
#             cur = get_variance_map(buf, patch_size, relative=True)
#         else:
#             print("Unexpected metric:", metric)
#
#         if importance_map is None:
#             importance_map = cur * weight
#         else:
#             importance_map += cur * weight
#
#     return importance_map / importance_map.max()
#
#
# def sample_patches(img_dim, patch_size, n_samples, maxiter=5000):
#     full_area = float(img_dim[0] * img_dim[1])
#     sample_area = full_area / n_samples
#     radius = np.sqrt(sample_area / np.pi)
#     minsqrdist = (2 * radius) ** 2
#
#     def get_sqrdist(x, y, patches):
#         if len(patches) == 0:
#             return np.infty
#         dist = patches - [x, y]
#         return np.sum(dist ** 2, axis=1).min()
#
#     # perform dart throwing, progressively reducing the radius
#     rate = 0.96
#     patches = np.zeros((n_samples, 2), dtype=int)
#     xmin, xmax = 0, img_dim[1] - patch_size[1] - 1
#     ymin, ymax = 0, img_dim[0] - patch_size[0] - 1
#     for patch in range(n_samples):
#         done = False
#         while not done:
#             for i in range(maxiter):
#                 x = random.randint(xmin, xmax)
#                 y = random.randint(ymin, ymax)
#                 sqrdist = get_sqrdist(x, y, patches[:patch, :])
#                 if sqrdist > minsqrdist:
#                     patches[patch, :] = [x, y]
#                     done = True
#                     break
#             if not done:
#                 radius *= rate
#                 minsqrdist = (2 * radius) ** 2
#
#     return patches
#
#
# def prune_patches(shape, patches, patchsize, imp):
#     pruned = np.empty_like(patches)
#
#     def get_regions_list(shape, step):
#         regions = []
#         for y in range(0, shape[0], step):
#             if y // step % 2 == 0:
#                 xrange = range(0, shape[1], step)
#             else:
#                 xrange = reversed(range(0, shape[1], step))
#             for x in xrange:
#                 regions.append((x, x + step, y, y + step))
#         return regions
#
#     def split_patches(patches, region):
#         cur = np.empty_like(patches)
#         rem = np.empty_like(patches)
#         ccount, rcount = 0, 0
#         for i in range(patches.shape[0]):
#             x, y = patches[i, 0], patches[i, 1]
#             if region[0] <= x < region[1] and region[2] <= y < region[3]:
#                 cur[ccount, :] = [x, y]
#                 ccount += 1
#             else:
#                 rem[rcount, :] = [x, y]
#                 rcount += 1
#         return cur[:ccount, :], rem[:rcount, :]
#
#     rem = np.copy(patches)
#     count, error = 0, 0
#     for region in get_regions_list(shape, 4 * patchsize):
#         cur, rem = split_patches(rem, region)
#         for i in range(cur.shape[0]):
#             x, y = cur[i, 0], cur[i, 1]
#             if imp[y, x] - error > random.random():
#                 pruned[count, :] = [x, y]
#                 count += 1
#                 error += 1 - imp[y, x]
#             else:
#                 error += 0 - imp[y, x]
#
#     return pruned[:count, :]
#
#
# def importanceSampling(data, patch_size=64, n_patches=200):
#     debug = False
#
#     # 1. extract buffers, using default and normal
#     buffers = []
#     for b in ["default", "normal"]:
#         buffers.append(data[b][:, :, :3])
#
#     # 2. build the metric map
#     metrics = ["relvar", "variance"]
#     weights = [1.0, 1.0]
#     imp = get_importance_map(buffers, metrics, weights, patch_size)
#
#     if debug:
#         print('Importance map:')
#         fig = plt.figure(figsize=(15, 15))
#         imgplot = plt.imshow(imp)
#         imgplot.axes.get_xaxis().set_visible(False)
#         imgplot.axes.get_yaxis().set_visible(False)
#         fig.savefig('assets/importance_map.png')
#
#     # 3. get patches
#     patches = sample_patches(buffers[0].shape[:2], (patch_size, patch_size), n_patches)
#
#     if debug:
#         print('Patches:')
#         fig = plt.figure(figsize=(15, 10))
#         plt.scatter(list(a[0] for a in patches), list(a[1] for a in patches))
#         fig.savefig('assets/patches.png')
#
#     # 4. get selection
#     selection = buffers[0] * 0.1
#     for i in range(patches.shape[0]):
#         x, y = patches[i, 0], patches[i, 1]
#         selection[y: y + patch_size, x: x + patch_size, :] = buffers[0][y: y + patch_size, x: x + patch_size, :]
#
#     # 5. prune patches
#     pad = patch_size // 2
#     pruned = np.maximum(0, prune_patches(buffers[0].shape[:2], patches + pad, patch_size, imp) - pad)
#     selection = buffers[0] * 0.1
#     for i in range(pruned.shape[0]):
#         x, y = pruned[i, 0], pruned[i, 1]
#         selection[y: y + patch_size, x: x + patch_size, :] = buffers[0][y: y + patch_size, x: x + patch_size, :]
#
#     if debug:
#         print('After pruning:')
#         fig = plt.figure(figsize=(15, 10))
#         plt.scatter(list(a[0] for a in pruned), list(a[1] for a in pruned))
#         fig.savefig('assets/pruned.png')
#
#     return pruned + pad

#########################################################
def getVarianceMap(data, patch_size, relative=False):
    # introduce a dummy third dimension if needed
    if data.ndim < 3:
        data = data[:, :, np.newaxis]

    # compute variance
    mean = ndimage.uniform_filter(data, size=(patch_size, patch_size, 1))
    sqrmean = ndimage.uniform_filter(data ** 2, size=(patch_size, patch_size, 1))
    variance = np.maximum(sqrmean - mean ** 2, 0)

    # convert to relative variance if requested
    if relative:
        variance = variance / np.maximum(mean ** 2, 1e-2)

    # take the max variance along the three channels, gamma correct it to get a
    # less peaky map, and normalize it to the range [0,1]
    variance = variance.max(axis=2)
    variance = np.minimum(variance ** (1.0 / 2.2), 1.0)

    return variance / variance.max()


# Generate importance sampling map based on buffer and desired metric
def getImportanceMap(buffers, metrics, weights, patch_size):
    if len(metrics) != len(buffers):
        metrics = [metrics[0]] * len(buffers)
    if len(weights) != len(buffers):
        weights = [weights[0]] * len(buffers)
    impMap = None
    for buf, metric, weight in zip(buffers, metrics, weights):
        if metric == 'uniform':
            cur = np.ones(buf.shape[:2], dtype=np.float)
        elif metric == 'variance':
            cur = getVarianceMap(buf, patch_size, relative=False)
        elif metric == 'relvar':
            cur = getVarianceMap(buf, patch_size, relative=True)
        else:
            print('Unexpected metric:', metric)
        if impMap is None:
            impMap = cur * weight
        else:
            impMap += cur * weight
    return impMap / impMap.max()


def samplePatchesProg(img_dim, patch_size, n_samples, maxiter=5000):
    # Sample patches using dart throwing (works well for sparse/non-overlapping patches)

    # estimate each sample patch area
    full_area = float(img_dim[0] * img_dim[1])
    sample_area = full_area / n_samples

    # get corresponding dart throwing radius
    radius = np.sqrt(sample_area / np.pi)
    minsqrdist = (2 * radius) ** 2

    # compute the distance to the closest patch
    def get_sqrdist(x, y, patches):
        if len(patches) == 0:
            return np.infty
        dist = patches - [x, y]
        return np.sum(dist ** 2, axis=1).min()

    # perform dart throwing, progressively reducing the radius
    rate = 0.96
    patches = np.zeros((n_samples, 2), dtype=int)
    xmin, xmax = 0, img_dim[1] - patch_size[1] - 1
    ymin, ymax = 0, img_dim[0] - patch_size[0] - 1
    for patch in range(n_samples):
        done = False
        while not done:
            for i in range(maxiter):
                x = random.randint(xmin, xmax)
                y = random.randint(ymin, ymax)
                sqrdist = get_sqrdist(x, y, patches[:patch, :])
                if sqrdist > minsqrdist:
                    patches[patch, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                minsqrdist = (2 * radius) ** 2

    return patches


def prunePatches(shape, patches, patchsize, imp):
    pruned = np.empty_like(patches)

    # Generate a set of regions tiling the image using snake ordering.
    def get_regions_list(shape, step):
        regions = []
        for y in range(0, shape[0], step):
            if y // step % 2 == 0:
                xrange = range(0, shape[1], step)
            else:
                xrange = reversed(range(0, shape[1], step))
            for x in xrange:
                regions.append((x, x + step, y, y + step))
        return regions

    # Split 'patches' in current and remaining sets, where 'cur' holds the
    # patches in the requested region, and 'rem' holds the remaining patches.
    def split_patches(patches, region):
        cur = np.empty_like(patches)
        rem = np.empty_like(patches)
        ccount, rcount = 0, 0
        for i in range(patches.shape[0]):
            x, y = patches[i, 0], patches[i, 1]
            if region[0] <= x < region[1] and region[2] <= y < region[3]:
                cur[ccount, :] = [x, y]
                ccount += 1
            else:
                rem[rcount, :] = [x, y]
                rcount += 1
        return cur[:ccount, :], rem[:rcount, :]

    # Process all patches, region by region, pruning them randomly according to
    # their importance value, ie. patches with low importance have a higher
    # chance of getting pruned. To offset the impact of the binary pruning
    # decision, we propagate the discretization error and take it into account
    # when pruning.
    rem = np.copy(patches)
    count, error = 0, 0
    for region in get_regions_list(shape, 4 * patchsize):
        cur, rem = split_patches(rem, region)
        for i in range(cur.shape[0]):
            x, y = cur[i, 0], cur[i, 1]
            if imp[y, x] - error > random.random():
                pruned[count, :] = [x, y]
                count += 1
                error += 1 - imp[y, x]
            else:
                error += 0 - imp[y, x]

    return pruned[:count, :]


def importanceSampling(data, patch_size=64, n_patches=200):
    debug = False

    # extract buffers
    buffers = []
    for b in ['default', 'normal']:
        buffers.append(data[b][:, :, :3])

    # build the metric map
    metrics = ['relvar', 'variance']
    weights = [1.0, 1.0]
    imp = getImportanceMap(buffers, metrics, weights, patch_size)

    if debug:
        print("Importance map:")
        plt.figure(figsize=(15, 15))
        imgplot = plt.imshow(imp)
        imgplot.axes.get_xaxis().set_visible(False)
        imgplot.axes.get_yaxis().set_visible(False)
        plt.show()

    # get patches
    patches = samplePatchesProg(buffers[0].shape[:2], (patch_size, patch_size), n_patches)

    if debug:
        print("Patches:")
        plt.figure(figsize=(15, 10))
        plt.scatter(list(a[0] for a in patches), list(a[1] for a in patches))
        plt.show()

    selection = buffers[0] * 0.1
    for i in range(patches.shape[0]):
        x, y = patches[i, 0], patches[i, 1]
        selection[y:y + patch_size, x:x + patch_size, :] = buffers[0][y:y + patch_size, x:x + patch_size, :]

    # prune patches
    pad = patch_size // 2
    pruned = np.maximum(0, prunePatches(buffers[0].shape[:2], patches + pad, patch_size, imp) - pad)
    selection = buffers[0] * 0.1
    for i in range(pruned.shape[0]):
        x, y = pruned[i, 0], pruned[i, 1]
        selection[y:y + patch_size, x:x + patch_size, :] = buffers[0][y:y + patch_size, x:x + patch_size, :]

    if debug:
        print("After pruning:")
        plt.figure(figsize=(15, 10))
        plt.scatter(list(a[0] for a in pruned), list(a[1] for a in pruned))
        plt.show()

    return pruned + pad


if __name__ == '__main__':
    file = pyexr.open('../renders/classroom/10499344-01024spp.exr')
    data = file.get_all()
    importanceSampling(data)
