import numpy as np
import os
import h5py
from tqdm import tqdm
from glob import glob
import pickle as pk
import json
import time
from scipy.spatial.distance import cdist
from future.utils import viewitems, lrange
from sklearn.metrics import precision_recall_curve
import faiss

def read_h5file(path):
    hf = h5py.File(path, 'r')
    g1 = hf.get('images')
    g2 = hf.get('names')
    return g1.keys(), g1, g2

def load_features(dataset_dir, is_gv=True):
    '''
    加载特征
    :param dataset_dir: 特征所在的目录, 例如：/home/camp/FIVR/features/vcms_v1
    :param is_gv: 是否取平均。True：返回帧平均的结果，False：保留所有帧的特征
    :return:
    '''
    h5_paths = glob(os.path.join(dataset_dir, '*.h5'))
    print(h5_paths)
    vfeat = {}
    vid2features = {}
    final_vids = []
    features = []
    for h5_path in h5_paths:
        vids, g1, g2 = read_h5file(h5_path)
        for vid in tqdm(vids):
            if is_gv:
                cur_arr = g1.get(vid)
                #print("1:",cur_arr.shape)
                cur_arr_ave = np.mean(cur_arr, axis=0, keepdims=False)
                cur_arr_max = np.max(cur_arr, axis=0, keepdims=False)
                cur_arr = np.concatenate([cur_arr_ave, cur_arr_max], axis=0)
                cur_arr /= (np.linalg.norm(cur_arr, ord=2, axis=0))
                #print(cur_arr.shape)
                vid2features[vid] = cur_arr
            else:
                cur_arr = g1.get(vid)
                #print("1:",cur_arr.shape)
                #cur_arr = np.concatenate([cur_arr, np.mean(cur_arr, axis=0, keepdims=True)], axis=0)
                cur_arr = np.asarray(cur_arr)
                cur_arr_mean = np.mean(cur_arr, axis=0, keepdims=True)
                vfeat[vid] = cur_arr_mean
                vid2features[vid] = cur_arr
                #print(cur_arr.shape)
                final_vids.extend([vid] * len(cur_arr))
                features.extend(cur_arr)
    if is_gv:
        return vid2features
    else:
        return final_vids, features, vfeat, vid2features

def calculate_similarities_matrix(query_features, all_features):
    """
      用于计算两组特征(已经做过l2-norm)之间的相似度
      Args:
        queries: shape: [N, D]
        features: shape: [M, D]
      Returns:
        similarities: shape: [N, M]
    """
    similarities = []
    # 计算待查询视频和所有视频的距离
    dist = np.nan_to_num(cdist(query_features, all_features, metric='cosine'))
    for i, v in enumerate(query_features):
        # 归一化，将距离转化成相似度
        # sim = np.round(1 - dist[i] / dist[i].max(), decimals=6)
        sim = 1-dist[i]
        # 按照相似度的从大到小排列，输出index
        similarities += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]]
    return similarities

def calculate_similarities(query_features, all_features):
    """
      用于计算两组特征(已经做过l2-norm)之间的相似度
      Args:
        queries: shape: [N, D]
        features: shape: [M, D]
      Returns:
        similarities: float
    """
    similarities = 0.0
    # 计算待查询视频和所有视频的距离
    dist = np.nan_to_num(cdist(query_features, all_features, metric='cosine'))
    sim = 1-dist
    similarities += np.sum(np.max(sim, axis = 1), axis=0)
    return similarities


top_k = 10000
def faiss_PQ(query_features, global_feattures):
    nlist = 100
    m = 8  # number of bytes per vector
    k = top_k
    d = query_features.shape[1]
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(global_feattures)
    index.add(global_feattures)
    index.nprobe = 20  # make comparable with experiment above
    now_time = time.time()
    D, I = index.search(query_features, k)  # search
    print("Retival time: ",time.time()-now_time)
    return D, I

def evaluateOfficial(annotations, results, relevant_labels, dataset, quiet):
    """
      Calculate of mAP and interpolated PR-curve based on the FIVR evaluation process.
      Args:
        annotations: the annotation labels for each query
        results: the similarities of each query with the videos in the dataset
        relevant_labels: labels that are considered positives
        dataset: video ids contained in the dataset
      Returns:
        mAP: the mean Average Precision
        ps_curve: the values of the PR-curve
    """
    pr, mAP = [], []
    iterations = viewitems(annotations) if not quiet else tqdm(viewitems(annotations))
    for query, gt_sets in iterations:
        query = str(query)
        if query not in results: print('WARNING: Query {} is missing from the result file'.format(query)); continue
        if query not in dataset: print('WARNING: Query {} is not in the dataset'.format(query)); continue

        # set of relevant videos
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt: print('WARNING: Empty annotation set for query {}'.format(query)); continue

        # calculation of mean Average Precision (Eq. 6)
        i, ri, s = 0.0, 0, 0.0
        y_target, y_score = [], []
        for video, sim in sorted(viewitems(results[query]), key=lambda x: x[1], reverse=True):
            if video in dataset:
                y_score.append(sim)
                y_target.append(1.0 if video in query_gt else 0.0)
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        mAP.append(s / len(query_gt))
        # add the dataset videos that are missing from the result file
        missing = len(query_gt) - y_target.count(1)
        y_target += [1.0 for _ in lrange(missing)] # add 1. for the relevant videos
        y_target += [0.0 for _ in lrange(len(dataset) - len(y_target))] # add 0. for the irrelevant videos
        y_score += [0.0 for _ in lrange(len(dataset) - len(y_score))]

        # calculation of interpolate PR-curve (Eq. 5)
        precision, recall, thresholds = precision_recall_curve(y_target, y_score)
        p = []
        for i in lrange(20, -1, -1):
            idx = np.where((recall >= i * 0.05))[0]
            p.append(np.max(precision[idx]))
        pr.append(p)
    # return mAP
    return mAP, np.mean(pr, axis=0)[::-1]

class GTOBJ:
    def __init__(self):
        annotation_path = '/home/camp/FIVR/annotation/annotation.json'
        dataset_path = '/home/camp/FIVR/annotation/youtube_ids.txt'
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        self.dataset = set(np.loadtxt(dataset_path, dtype=str).tolist())
gtobj = GTOBJ()
relevant_labels_mapping = {
    'DSVR': ['ND','DS'],
    'CSVR': ['ND','DS','CS'],
    'ISVR': ['ND','DS','CS','IS'],
}

now_time = time.time()
final_vids, features, vfeat, vid2features = load_features('/home/camp/FIVR/features/vcms_v1', is_gv=False)
print("loading time: ",time.time()-now_time)


# 加载特征
vids = list(vid2features.keys())
print(vids[:10])

global_feattures = [np.asarray(i,np.float32) for i in list(vid2features.values())]

global_frame_feattures = np.asarray(features)
print("global_frame_featture_shape: ", global_frame_feattures.shape)

# 加载vid2name 和 name2vid
with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    vid2names = pk.load(pk_file)
with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    name2vids = pk.load(pk_file)


# 开始评估
annotation_dir = '/home/camp/FIVR/annotation'
names = np.asarray([vid2names[vid][0] for vid in vids])
query_names = None
results = None

for task_name in ['DSVR', 'CSVR', 'ISVR']:
    annotation_path = os.path.join(annotation_dir, task_name + '.json')
    with open(annotation_path, 'r') as annotation_file:
        json_obj = json.load(annotation_file)
    if results is None:
        query_names = json_obj.keys()
        query_names = [str(query_name) for query_name in query_names]
        query_indexs = []
        print("query len:", len(query_names))
        for query_name in query_names:
            tmp = np.where(names == query_name)
            if len(tmp) != 0 and len(tmp[0]) != 0:
                query_indexs.append(tmp[0][0])
            else:
                print('skip query: ', query_name)

        num_global_frames = global_frame_feattures.shape[0]

        query_feat = [global_feattures[idx] for idx in query_indexs]
        print(len(query_feat))
        query_features = np.concatenate([global_feattures[idx] for idx in query_indexs], axis=0)
        print("query_features_shape: ",query_features.shape)

        results = dict()
        vis = dict()
        
        # now_time = time.time()
        D, I = faiss_PQ(query_features,global_frame_feattures)
        print("D_firstrow_10: ",D[0,:10])
        # print("Retival time: ",time.time() - now_time)
        print("D_shape: ",D.shape)
        print("I_shape: ",I.shape)
        now_time = time.time()
        start = 0
        for _,id in enumerate(tqdm(query_indexs)):
            query_name = query_names[_]
            print("vedio id: ",query_name)
            length = global_feattures[id].shape[0]
            ending = start + length
            sim = dict()
            for iter_x in range(start, ending):
                for iter_y in range(top_k):
                    now_score = D[iter_x][iter_y]
                    cur_video_id = vid2names[final_vids[ I[iter_x][iter_y] ]][0]
                    if cur_video_id in vis and vis[cur_video_id] == iter_x:
                        continue
                    else:   
                        if not cur_video_id in sim:
                            sim[cur_video_id] = 0.0
                        sim[cur_video_id] += now_score
                    vis[cur_video_id] = iter_x
            start = ending
            if query_name in sim:
                del sim[query_name]
            results[query_name] = sim
        print("Process time: ",time.time()-now_time)
    # print(results[query_names[0]])
    mAPOffcial, precisions = evaluateOfficial(annotations=gtobj.annotations, results=results,
                                                  relevant_labels=relevant_labels_mapping[task_name],
                                                  dataset=gtobj.dataset,
                                                  quiet=False)
    print('{} mAPOffcial is {}'.format(task_name, np.mean(mAPOffcial)))


