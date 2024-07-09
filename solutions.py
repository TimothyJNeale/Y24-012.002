
import numpy as np
#
# Solution functions from 'the first 10 tasks' notebook
# https://www.kaggle.com/code/nagiss/manual-coding-for-the-first-10-tasks
#

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}

def task_train000(x):
    x_upsampled = x.repeat(3, axis=0).repeat(3, axis=1)
    x_tiled = np.tile(x, (3, 3))
    y = x_upsampled & x_tiled
    return y

def task_train001(x):
    green, yellow = color2num["green"], color2num["yellow"]
    
    def get_closed_area(arr):
        # depth first search
        H, W = arr.shape
        Dy = [0, -1, 0, 1]
        Dx = [1, 0, -1, 0]
        arr_padded = np.pad(arr, ((1,1),(1,1)), "constant", constant_values=0)
        searched = np.zeros(arr_padded.shape, dtype=bool)
        searched[0, 0] = True
        q = [(0, 0)]
        while q:
            y, x = q.pop()
            for dy, dx in zip(Dy, Dx):
                y_, x_ = y+dy, x+dx
                if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                    continue
                if not searched[y_][x_] and arr_padded[y_][x_]==0:
                    q.append((y_, x_))
                    searched[y_, x_] = True
        res = searched[1:-1, 1:-1]
        res |= arr==green
        return ~res
        
    y = x.copy()
    y[get_closed_area(x)] = yellow
    return y

def task_train002(x):
    red, blue = color2num["red"], color2num["blue"]
    
    def get_period_length(arr):
        H, W = arr.shape
        period = 1
        while True:
            cycled = np.pad(arr[:period, :], ((0,H-period),(0,0)), 'wrap')
            if (cycled==arr).all():
                return period
            period += 1
            
    def change_color(arr, d):
        res = arr.copy()
        for k, v in d.items():
            res[arr==k] = v
        return res
            
    period = get_period_length(x)
    y = x[:period, :]  # clop one period
    y = np.pad(y, ((0,9-period),(0,0)), 'wrap')  # cycle
    y = change_color(y, {blue: red})
    return y

# This next one written by me with help form LLM
def task_train003(x):
    # The solution would detect where there is a shape on the task pllot, 
    # then slie the top rows of the shape only one step to the rgi

    # Find the top row of each shape
    rows_to_move = []
    shapes = []
    in_shape = False
    # for every row in the task
    for i in range(x.shape[0]):

        # Is there any colour on that row
        if x[i].sum()>0:
            # Mark such a row to shift right
            rows_to_move.append(i)
            in_shape = True
    
        else:
            # if we no longer in a shape but the in shape flag is set, the do not shift the previous row
            if in_shape:
                rows_to_move.pop()
                in_shape = False
                shapes.append(rows_to_move)

    # Slide the top row to the right, 
    y = x.copy()

    # for each shape find the furthest rightward coloured cell
    for shape in shapes:
        imax = 0
        for row in shape:
            imax = max(imax, np.where(x[row]>0)[0][-1])

        # now all the cells int he reow that should be moved to the right, but no bigher than imax
        for row in shape:
            # shift all the cells in that row to the right but not if it is the the furthest right coloured cell
            for i in range(x.shape[1]-1, 0, -1):
                if x[row, i]>0 and i < imax:
                    y[row, i+1] = x[row, i]
                    y[row, i] = 0


    return y

def task_train004(x):
    def get_3x3_base_pattern(arr):
        # find maximum number of unique color tiles in 3x3 field
        H, W = arr.shape
        arr_onehot = 1<<arr
        arr_bool = arr.astype(bool).astype(np.int32)
        counts = np.zeros(arr.shape, dtype=np.int32)
        colors = np.zeros(arr.shape, dtype=np.int32)
        for y in range(H-2):
            for x in range(W-2):
                counts[y, x] = arr_bool[y:y+2, x:x+2].sum()
                colors[y, x] = np.bitwise_or.reduce(arr_onehot[y:y+2, x:x+2].reshape(-1))
        n_colors = np.zeros(arr.shape, dtype=np.int32)
        for c in range(1, 10):
            n_colors += colors>>c & 1
        counts[n_colors>=2] = 0
        res_y, res_x = np.unravel_index(np.argmax(counts), counts.shape)
        pattern = arr[res_y:res_y+3, res_x:res_x+3].astype(bool).astype(np.int32)
        return (res_y, res_x), pattern
    
    (base_y, base_x), pattern = get_3x3_base_pattern(x)
    pad_size = 25
    x_padded = np.pad(x, ((pad_size,pad_size),(pad_size,pad_size)), "constant", constant_values=0)
    base_y += pad_size
    base_x += pad_size
    y = x_padded.copy()
    for dy in [-4, 0, 4]:
        for dx in [-4, 0, 4]:
            if dy==dx==0:
                continue
            y_, x_ = base_y+dy, base_x+dx
            count = np.bincount(x_padded[y_:y_+4, x_:x_+4].reshape(-1))
            if count[0]==9:
                continue
            count[0] = 0
            color = count.argmax()
            for i in range(1, 6):
                # repeat pattern
                y[base_y+dy*i:base_y+dy*i+3, base_x+dx*i:base_x+dx*i+3] = color * pattern
    y = y[pad_size:-pad_size, pad_size:-pad_size]
    return y

def task_train005(x):
    blue, red = color2num["blue"], color2num["red"]
    
    def split_by_gray_line(arr):
        H, W = arr.shape
        gray = color2num["gray"]
        Y = [-1]
        for y in range(H):
            if (arr[y, :]==gray).all():
                Y.append(y)
        Y.append(H)
        X = [-1]
        for x in range(W):
            if (arr[:, x]==gray).all():
                X.append(x)
        X.append(W)
        res = [[arr[y1+1:y2, x1+1:x2] for x1, x2 in zip(X[:-1], X[1:])] for y1, y2 in zip(Y[:-1], Y[1:])]
        return res
    
    def change_color(arr, d):
        res = arr.copy()
        for k, v in d.items():
            res[arr==k] = v
        return res
            
    x_split = split_by_gray_line(x)
    assert len(x_split)==1
    assert len(x_split[0])==2
    x1, x2 = x_split[0]
    y = x1 & x2
    y = change_color(y, {blue: red})
    return y

def task_train006(x):
    H, W = x.shape
    colors = [0, 0, 0]
    for yy in range(H):
        for xx in range(W):
            color = x[yy, xx]
            if color != 0:
                colors[(yy+xx)%3] = color
    y = x.copy()
    for yy in range(H):
        for xx in range(W):
            y[yy, xx] = colors[(yy+xx)%3]
    return y


def task_train007(x):
    sky, red = color2num["sky"], color2num["red"]
    square_idx_set = set(tuple(idx) for idx in np.array(np.where(x==sky)).T)
    object_idx_list = [tuple(idx) for idx in np.array(np.where(x==red)).T]
    Dy = [0, 1, 0, -1]
    Dx = [1, 0, -1, 0]
    for dy, dx in zip(Dy, Dx):
        for n in range(1, 100):
            obj_idx = set((idx[0]+dy*n, idx[1]+dx*n) for idx in object_idx_list)
            if obj_idx & square_idx_set:
                y = np.zeros(x.shape, dtype=np.int32)
                for idx in square_idx_set:
                    y[idx] = sky
                for idx in obj_idx:
                    idx = (idx[0]-dy, idx[1]-dx)
                    y[idx] = red
                return y
    assert False


def task_train008(x):
    H, W = x.shape
    y = x.copy()
    l, r = x.copy(), x.copy()
    for yy in range(H):
        for xx in range(3, W):
            if x[yy, xx] == 0:
                l[yy, xx] = l[yy, xx-3]
        for xx in range(W-4, -1, -1):
            if x[yy, xx] == 0:
                r[yy, xx] = r[yy, xx+3]
        for xx in range(W):
            if l[yy, xx] == r[yy, xx]:
                y[yy, xx] = l[yy, xx]
    u, d = x.copy(), x.copy()
    for xx in range(W):
        for yy in range(3, H):
            if x[yy, xx] == 0:
                u[yy, xx] = u[yy-3, xx]
        for yy in range(H-4, -1, -1):
            if x[yy, xx] == 0:
                d[yy, xx] = d[yy+3, xx]
        for yy in range(H):
            if u[yy, xx] == d[yy, xx]:
                y[yy, xx] |= u[yy, xx]  # ignore black tiles by using '|='
    return y


def task_train009(x):
    H, W = x.shape
    y = x.copy()
    gray, blue, red, green, yellow = color2num["gray"], color2num["blue"], color2num["red"], color2num["green"], color2num["yellow"]
    colors = [blue, red, green, yellow]
    colors_idx = 0
    for yy in range(H):
        for xx in range(W):
            if y[yy, xx]==gray:
                for y_ in range(yy, H):
                    y[y_, xx] = colors[colors_idx]
                colors_idx += 1
    return y