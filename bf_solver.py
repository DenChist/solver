import itertools
import numpy as np


def gen_masks(n, m):
    comb = np.array(list(itertools.combinations(range(1, n), m - 1)), int)
    a = np.hstack((np.zeros((comb.shape[0], 1), int), comb))
    b = np.hstack((comb, np.full((comb.shape[0], 1), n, int)))
    return np.unique(np.sort(b - a), axis=0)

def gen_class_data(mask):
    max_num = mask[-1]
    class_list = np.unique(mask)
    class_count = np.zeros(max_num + 1, int)
    for num in mask:
        class_count[num] += 1
    class_index = np.zeros(max_num + 1, int)
    sum = 0
    for num in class_list:
        class_index[num] = sum
        sum += class_count[num] * num
    return class_list, class_count[class_list], class_index[class_list]

def gen_confs(perm, masks):
    res_list = list()
    for mask in masks:
        class_num, class_count, class_index = gen_class_data(mask)
        list_array = np.empty(perm.shape, int)
        for num, count, index in zip(class_num, class_count, class_index):
            cur_slice = slice(index, index + count * num)
            cur_shape = (perm.shape[0], count, num)
            perm_slice = perm[:, cur_slice]
            perm_reshape = perm_slice.reshape(cur_shape)
            list_slice = list_array[:, cur_slice]
            list_reshape = list_slice.reshape(cur_shape)
            if count > 1:
                for lr, pr in zip(list_reshape, perm_reshape):
                    lr[:] = pr[pr[:, 0].argsort()]
            else:
                list_reshape[:] = perm_reshape[:]
        list_un = np.unique(list_array, axis=0)
        res_list.append(list_un)
    return res_list

class StageSolver:
    def __init__(self, masks, confs, stage_number, st, ct, p):
        self.m_masks = masks
        self.m_confs = confs
        self.m_mask_slice = self.gen_mask_slice()
        self.m_slice_list = self.gen_slice_list()
        self.m_mask_count = len(masks)
        self.m_stage_number = stage_number
        self.m_prev_ct_line = np.zeros(ct.shape[1])
        if stage_number > 0:
            self.m_prev_ct_line = ct[stage_number - 1, :]
        self.m_st_line = st[stage_number, :]
        self.m_ct_line = ct[stage_number, :]
        self.m_p_line = p[stage_number, :]

        self.m_cur_index = 0
        self.m_cur_conf_len = len(confs[0])
        self.m_cur_pos = 0
        self.m_is_end = False
        self.m_cur_slise_list = list()
        self.set_cur_slice()

    def gen_mask_slice(self):
        res = list()
        for mask in self.m_masks:
            start = np.zeros(len(mask), int)
            end = np.copy(mask)
            for i in range(1, len(mask)):
                start[i] = end[i - 1]
                end[i] += end[i - 1]
            slice_list = list()
            for s, e in zip(start, end):
                slice_list.append(slice(s, e))
            res.append(slice_list)
        return res
    
    def gen_slice_list(self):
        res = list()
        for m_s, conf in zip(self.m_mask_slice, self.m_confs):
            slice_list = list()
            for sl in m_s:
                slice_list.append(conf[:, sl])
            res.append(slice_list)
        return res
    
    def set_cur_slice(self):
        self.m_cur_slise_list = self.m_slice_list[self.m_cur_index]

    def get(self):
        return self.m_masks[self.m_cur_index], self.m_confs[self.m_cur_index][self.m_cur_pos]
    
    def calc(self):
        for jobs_on_machine in self.m_cur_slise_list:
            prev_job_ct = 0
            for job in jobs_on_machine[self.m_cur_pos]:
                self.m_st_line[job] = max(prev_job_ct, self.m_prev_ct_line[job])
                self.m_ct_line[job] = self.m_st_line[job] + self.m_p_line[job]
                prev_job_ct = self.m_ct_line[job]

    def first(self):
        self.m_cur_index = 0
        self.m_cur_conf_len = len(self.m_confs[0])
        self.m_cur_pos = 0
        self.m_is_end = False
        self.set_cur_slice()

    def next(self):
        self.m_cur_pos += 1
        if self.m_cur_pos >= self.m_cur_conf_len:
            self.m_cur_pos = 0
            self.m_cur_index += 1
            if self.m_cur_index >= self.m_mask_count:
                self.m_cur_index = 0
                self.m_is_end = True
            self.set_cur_slice()
            self.m_cur_conf_len = len(self.m_confs[self.m_cur_index])

    def is_end(self):
        return self.m_is_end

def solver(n, s, p):
    perm = np.array(list(itertools.permutations(range(0, n))), int)
    masks = dict()
    confs = dict()
    st = np.zeros((len(s), n))
    ct = np.zeros((len(s), n))
    end_line_slice = ct[-1, :]
    stage_solver_list = list()
    for stage_number, machine_count in enumerate(s):
        if not machine_count in masks:
            masks[machine_count] = gen_masks(n ,machine_count)
            confs[machine_count] = gen_confs(perm, masks[machine_count])
        stage_solver_list.append(StageSolver(masks[machine_count], confs[machine_count], stage_number, st, ct, p))

    res_min = 1000000000
    res_ct = np.zeros((len(s), n))
    res_masks = list()
    res_confs = list()
    stop = False
    while not stop:
        for s_solver in stage_solver_list:
            s_solver.calc()
        cur_res = max(end_line_slice)
        if cur_res < res_min:
            res_min = cur_res
            print(res_min)
            res_ct = np.copy(ct)
            res_cur_masks = list()
            res_cur_confs = list()
            for s_solver in stage_solver_list:
                mask, conf = s_solver.get()
                res_cur_masks.append(mask)
                res_cur_confs.append(conf)
            res_masks = res_cur_masks
            res_confs = res_cur_confs

        prev_stage_is_end = True
        index = 0
        while prev_stage_is_end:
            if index < len(stage_solver_list):
                stage_solver_list[index].next()
                if stage_solver_list[index].is_end():
                    stage_solver_list[index].first()
                else:
                    prev_stage_is_end = False
                index += 1
            else:
                stop = True
                break

    return res_min, res_ct, res_masks, res_confs

def main():
    n = 5
    s = np.array([3, 2, 3])
    p = np.array([[20, 15, 30, 25, 10],
                  [30, 20, 15, 10, 25],
                  [10, 25, 20, 30, 15]])

    res_min, res_ct, res_masks, res_confs = solver(n, s, p)
    print(res_min)
    print(res_ct)
    print()
    for stage, (mask, conf) in enumerate(zip(res_masks, res_confs)):
        print(f"stage: {stage} mask: {mask} conf: {conf}")
 
if __name__ == "__main__":
    main()