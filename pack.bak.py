import csv
import random
from itertools import permutations
sleigh_size = [1000, 1000]

# XXX how to orient presents once I found a slot?

class Node:
    def __init__(self):
        self.left, self.right = None, None
        self.filled = False
        self.dim, self.val = None, None
        self.coord = [-1, -1]
        self.size = [-1, -1]

class Tree:
    def __init__(self):
        self.root = Node()
        self.root.coord = [0, 0]
        self.root.size = sleigh_size

    def insert_present(self, node, present):

        # best:
        # GUILLOTINE-BSSF-SAS-RM-DESCSS-BFF 

        # make sure it's a leaf
        #assert node.left == None
        #assert node.right == None
        
        # make sure it fits
        #assert present[0] <= (sleigh_size[0]-node.coord[0])
        #assert present[1] <= (sleigh_size[1]-node.coord[1])
        
        #if (sleigh_size[0]-present[0]) < (sleigh_size[1]-present[1]):
        if node.size[0] > node.size[1]:
            node.dim = 0
        else:
            node.dim = 1
        
        # split first dim
        node.val = node.coord[node.dim] + present[node.dim]
        node.right = Node()
        node.right.coord[node.dim] = node.val
        node.right.coord[(node.dim+1)%2] = node.coord[(node.dim+1)%2]
        node.right.size[node.dim] = node.size[node.dim] + node.coord[node.dim] - node.val 
        node.right.size[(node.dim+1)%2] = node.size[(node.dim+1)%2]
        node.left = Node()
        node.left.coord = node.coord[:]
        node.left.size[node.dim] = node.val - node.coord[node.dim]
        node.left.size[(node.dim+1)%2] = node.size[(node.dim+1)%2]
        
        #assert (node.left.size[node.dim] + node.right.size[node.dim]) == node.size[node.dim]
        #assert node.left.size[(node.dim+1)%2] == node.right.size[(node.dim+1)%2] == node.size[(node.dim+1)%2]
        
        # split next dim
        node.left.dim = (node.dim+1)%2
        node.left.val = node.left.coord[node.left.dim] + present[node.left.dim]
        node.left.right = Node()
        node.left.right.coord[node.left.dim] = node.left.val
        node.left.right.coord[node.dim] = node.left.coord[node.dim]
        node.left.right.size[node.left.dim] = node.left.size[node.left.dim] + node.left.coord[node.left.dim] - node.left.val
        node.left.right.size[node.dim] = node.left.size[node.dim]
        node.left.left = Node()
        node.left.left.coord = node.left.coord[:]
        node.left.left.size[node.dim] = node.left.size[node.dim]
        node.left.left.size[node.left.dim] = node.left.val - node.left.coord[node.left.dim]
        node.left.left.filled = True
        
        #assert (node.left.left.size[node.left.dim] + node.left.right.size[node.left.dim]) == node.left.size[node.left.dim]
        #assert node.left.left.size[(node.left.dim+1)%2] == node.left.right.size[(node.left.dim+1)%2] == node.left.size[(node.left.dim+1)%2]

    def serach_tree(self, present):
        best_density = 0.
        best_node = None
        
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            if not node.filled:
                # check if it fits
                if (present[0] < node.size[0]) and (present[1] < node.size[1]):
                    if node.left == None:
                        #assert node.right == None
                        density = float(present[0]*present[1])/float(node.size[0]*node.size[1])
                        if density > best_density:
                            best_density = density
                            best_node = node
                    else:
                        stack.append(node.left)
                        stack.append(node.right)
        return best_node, best_density

def make_submission(presents, coords, filename='submission.csv'):
    
    with open(filename, 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['PresentId',
                         'x1','y1','z1','x2','y2','z2',
                         'x3','y3','z3','x4','y4','z4',
                         'x5','y5','z5','x6','y6','z6',
                         'x7','y7','z7','x8','y8','z8'])
        for pid in range(len(presents)):
            assert coords[pid][0] >= 0
            assert coords[pid][1] >= 0
            assert coords[pid][2] >= 0
            writer.writerow([pid+1, 
                             coords[pid][0] + 1, coords[pid][1] + 1, coords[pid][2] + 1,
                             coords[pid][0] + presents[pid][0], coords[pid][1] + 1, coords[pid][2] + 1,
                             coords[pid][0] + 1, coords[pid][1] + presents[pid][1], coords[pid][2] + 1,
                             coords[pid][0] + presents[pid][0], coords[pid][1] + presents[pid][1], coords[pid][2] + 1,
                             coords[pid][0] + 1, coords[pid][1] + 1, coords[pid][2] + presents[pid][2],
                             coords[pid][0] + presents[pid][0], coords[pid][1] + 1, coords[pid][2] + presents[pid][2],
                             coords[pid][0] + 1, coords[pid][1] + presents[pid][1], coords[pid][2] + presents[pid][2],
                             coords[pid][0] + presents[pid][0], coords[pid][1] + presents[pid][1], coords[pid][2] + presents[pid][2]])

def flip(pres, i0, i1):
    pres[i0], pres[i1] = pres[i1], pres[i0]

def pack_layer(pres_layer):
    Nlayer = len(pres_layer)
    packed = [False]*Nlayer
    coords_layer = [[-1,-1,-1]]*Nlayer

    pid_max, zmax = -1, -1
    for pid, x in enumerate(pres_layer):
        if min(x) > zmax:
            zmax = min(x)
            pid_max = pid
            
    # flip the orientation of the first present
    pres_layer[pid_max] = sorted(pres_layer[pid_max], reverse=True)
    assert pres_layer[pid_max][2] == zmax
    coords_layer[pid_max] = [0,0,layer_x3]
    packed[pid_max] = True                

    # then flip the orientations of all remaining presents (suboptimal?)
    
    for pid in [i for i in range(Nlayer) if i != pid_max]:
        pres_layer[pid] = sorted(pres_layer[pid])
        if pres_layer[pid][2] > zmax:
            if pres_layer[pid][1] > zmax:
                flip(pres_layer[pid], 0, 2)
            elif pres_layer[pid][0] > zmax:
                flip(pres_layer[pid], 1, 2)
            else:
                if pres_layer[pid][0] > pres_layer[pid][1]:
                    flip(pres_layer[pid], 0, 2)
                else:
                    flip(pres_layer[pid], 1, 2)

        assert pres_layer[pid][2] <= zmax
    
    # now the real work begins...
    tree = Tree()
    tree.insert_present(tree.root, pres_layer[pid_max])
    for idx in range(Nlayer-1):
        max_density = 0.
        max_node, max_pid = None, None
        for pid in range(Nlayer):
            if not packed[pid]:
                node0, density0 = tree.serach_tree(pres_layer[pid])
                node1, density1 = tree.serach_tree([pres_layer[pid][1], pres_layer[pid][0], pres_layer[pid][2]])
                if max(density1, density0) > max_density:
                    if density1 > density0:
                        pres_layer[pid] = [pres_layer[pid][1], pres_layer[pid][0], pres_layer[pid][2]]
                        max_density, max_node = density1, node1
                    else:
                        max_density, max_node = density0, node0
                    max_pid = pid

        if max_node == None:
            return None
        
        #print max_density, max_pid, pres_layer[max_pid]
        tree.insert_present(max_node, pres_layer[max_pid])
        packed[max_pid] = True
        coords_layer[max_pid] = [max_node.coord[0], max_node.coord[1], layer_x3]
                
    return pres_layer, coords_layer

def fits(pres0, pres1, zmax):
    if (pres1[2] + pres0[2]) <= zmax:
        if pres0[0] <= pres1[0]:
            if pres0[1] <= pres1[1]:
                return float(pres0[0]*pres0[1]*pres0[2])/float(pres1[0]*pres1[1]*(zmax-pres1[2]))
    return -1.0

def pack_z(pres_layer, coords_layer, packed):
    #0.6627 2432461
    #0.6651 2423472
    #0.6869 2346747
    #0.6930 2325995
    #0.6986 2307339
    #0.7013 2298562
    #0.7235 2227944
    
    Nlayer = len(pres_layer)
    
    pres_layer2 = [pres_layer[pid] for pid in range(Nlayer) if packed[pid]]
    coords_layer2 = [coords_layer[pid] for pid in range(Nlayer) if packed[pid]]
    
    zmax = max([pres_layer[pid][2] for pid in range(Nlayer)])
    
    # just check if everyone can be lazily packed on top of someone else
    for pid in range(Nlayer):
        if not packed[pid]:
            max_density, max_pres, max_zid = 0., None, None
            for zid in range(len(pres_layer2)):
                for idx0, idx1, idx2 in permutations(range(3), 3):
                    pres = [pres_layer[pid][idx0], pres_layer[pid][idx1], pres_layer[pid][idx2]]
                    density = fits(pres, pres_layer2[zid], zmax)
                    if density > max_density:
                        max_density = density
                        max_pres = pres[:]
                        max_zid = zid
            
            if max_zid is not None:
                packed[pid] = True
                pres_layer[pid] = max_pres[:]
                coords_layer[pid] = [coords_layer2[max_zid][0], coords_layer2[max_zid][1], zmax-pres_layer[pid][2]+coords_layer2[max_zid][2]]
                
                # add sub rectangles (similar to tree algo)
                if (pres_layer2[max_zid][0] - pres_layer[pid][0]) > (pres_layer2[max_zid][1] - pres_layer[pid][1]):
                    pres_layer2.append([pres_layer2[max_zid][0] - pres_layer[pid][0], pres_layer2[max_zid][1], pres_layer2[max_zid][2]])
                    coords_layer2.append([coords_layer2[max_zid][0] + pres_layer[pid][0], coords_layer2[max_zid][1], coords_layer2[max_zid][2]])
                    
                    pres_layer2.append([pres_layer[pid][0], pres_layer2[max_zid][1] - pres_layer[pid][1], pres_layer2[max_zid][2]])
                    coords_layer2.append([coords_layer2[max_zid][0], coords_layer2[max_zid][1] + pres_layer[pid][1], coords_layer2[max_zid][2]])
                else:
                    pres_layer2.append([pres_layer2[max_zid][0], pres_layer2[max_zid][1] - pres_layer[pid][1], pres_layer2[max_zid][2]])
                    coords_layer2.append([coords_layer2[max_zid][0], coords_layer2[max_zid][1] + pres_layer[pid][1], coords_layer2[max_zid][2]])
                    
                    pres_layer2.append([pres_layer[pid][0], pres_layer[pid][1], pres_layer2[max_zid][2]])
                    coords_layer2.append([coords_layer2[max_zid][0] + pres_layer[pid][0], coords_layer2[max_zid][1], coords_layer2[max_zid][2]])

                del coords_layer2[max_zid], pres_layer2[max_zid]
            else:
                break
    
    if sum(packed) < Nlayer:
        return None
    else:
        return pres_layer, coords_layer
                

    return None 

    Nlayer = len(pres_layer)
    
    # first we make intermediate datastructures
    # sort packed presents by z value
    z_sorted = sorted([(idx, coords_layer[idx][2] + pres_layer[idx][2]) for idx in range(Nlayer)], key=lambda tup: tup[1])
    
    zmax = z_sorted[-1][1]
    
    # build up list of lists containing indicies of all adjacent packed presents with z > myz (i.e. where there could be a conflict)
    # of course this only cuts the search space in half...

    adjacent = [[] for pid in range(Nlayer)]
    for pid in range(Nlayer):
        if packed[pid]:
            for aid in range(Nlayer):
                if packed[aid]:
                    if (coords_layer[aid][2] + pres_layer[aid][2] > coords_layer[pid][2] + pres_layer[pid][2]):
                        if (coords_layer[aid][0] - coords_layer[pid][0] - pres_layer[pid][0]) <= 250:
                            adjacent[pid] += [aid]
                        elif (coords_layer[aid][1] - coords_layer[pid][1] - pres_layer[pid][1]) <= 250:
                            adjacent[pid] += [aid]
    
    x_sorted = sorted([(idx, coords_layer[idx][0] + pres_layer[idx][0]) for idx in range(Nlayer)], key=lambda tup: tup[1])
    y_sorted = sorted([(idx, coords_layer[idx][1] + pres_layer[idx][1]) for idx in range(Nlayer)], key=lambda tup: tup[1])
    
    # also keep track of who has a present on top of them
    top_packed = [False]*Nlayer
    
    for pid in range(Nlayer):
        if not packed[pid]:
            # search top down for a place to pack
            for zid, z in z_sorted:
                if (pres_layer[pid][2] + z) < zmax:
                    # all wrong...
                    adj_zmax = z
                    for aid in adjacent[zid]:
                        adj_zmax = max(adj_zmax, coords_layer[aid][2] + pres_layer[aid][2])
                    if (pres_layer[pid][2] + adj_zmax) < zmax:
                        import ipdb; ipdb.set_trace()
                else:
                    # nowhere to go, have to return None
                    return None


presents = []
with open('presents.csv', 'r') as fd:
    reader = csv.reader(fd)
    reader.next()
    for pid, x1, x2, x3 in reader:
        presents.append([int(x1),int(x2),int(x3)])
#presents = presents[:1000]

Np = len(presents)

# each box can be packed by (x1, x2, x3) coordinates
# we modify presents array to indicate final orientation
# (submission file produced after the fact...)

coords = [[-1,-1,-1]]*Np   # (luckily this actually copies the object and not the reference...)

# build up layers one-by-one
layer_x3 = 0
pid_current = 0
avg_ratio = 0
while True:
    Nlayer = 2
    # first find upper bound
    while (pack_layer(presents[pid_current:(pid_current+Nlayer)])):
        Nlayer *= 2
    
    Nupper = Nlayer
    Nlower = Nlayer/2

    while (Nupper - Nlower > 1):
        Nlayer = (Nupper + Nlower)/2
        if (pack_layer(presents[pid_current:(pid_current+Nlayer)])):
            Nlower = Nlayer
        else:
            Nupper = Nlayer
            
    Nlayer = Nlower    
    pres_layer, coords_layer = pack_layer(presents[pid_current:(pid_current+Nlayer)])
    
    presents[pid_current:pid_current+len(pres_layer)] = pres_layer
    coords[pid_current:pid_current+len(pres_layer)] = coords_layer
    pid_current += Nlayer
    layer_height = max([x[2] for x in pres_layer[:Nlayer]])
    layer_x3 += layer_height
    
    while (pack_layer(presents[pid_current:(pid_current+Nlayer)])):
        Nlayer *= 2
    
    Nupper = Nlayer
    Nlower = Nlayer/2

    while (Nupper - Nlower > 1):
        Nlayer = (Nupper + Nlower)/2
        if (pack_layer(presents[pid_current:(pid_current+Nlayer)])):
            Nlower = Nlayer
        else:
            Nupper = Nlayer

    

    if pid_current >= 50000: #Np:
        break
    else:
        Nbuff = 100
        pres_buff = presents[pid_current:(pid_current+Nbuff)]
                
        # store placed presents sorted according to corners
        #coords_layer = sorted(zip(coords_layer, range(Nlayer)), key=lambda tup: tup[0][0]+tup[0][1])
        
        for idx in range(Nlayer):
            coords_layer[idx][2] += pres_layer[idx][2]
        '''
        while True:
            best_h, best_idx = None, None
            for idx in range(Nlayer):
                h = coords_layer[idx][2] + presents[pid_current][2]
                if h < layer_height:                    
                    for jdx in [i for i in range(Nlayer) if i != idx]:
                        if coords_layer[jdx][0] > coords_layer[idx][0] and coords_layer[jdx][1] > coords_layer[idx][1]:
                            if coords_layer[jdx][0] < (coords_layer[idx][0] + presents[pid_current][0]):
                                if coords_layer[jdx][1] < (coords_layer[idx][1] + presents[pid_current][1]):
                                    h = min(h, coords_layer[jdx][2] + presents[pid_current][2])
                                    if h > layer_height:
                                        break
                                    if h > best_h:
                                        best_h = h
                                        best_idx = idx
                                        
                    
            break
        break
        '''
        
    '''
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    left, bottom, width, height, color = [], [], [], [], []
    for pid in range(Nlayer):
        left.append(coords_layer[pid][0])
        bottom.append(coords_layer[pid][1])
        width.append(pres_layer[pid][0])
        height.append(pres_layer[pid][1])
        color.append((random.random(), random.random(), random.random(), 0.5))
    plt.bar(left=left, bottom=bottom, width=width, height=height, color=color)
    plt.xlim(0,sleigh_size[0])
    plt.ylim(0,sleigh_size[1])        
    break
    '''

    packing_ratio = sum(pres_layer[pid][0]*pres_layer[pid][1]*pres_layer[pid][2] for pid in range(Nlayer))/float(1000*1000*layer_height)
    avg_ratio += packing_ratio*Nlayer
    print packing_ratio, avg_ratio/float(pid_current), 806030653890/(1000*1000)/(avg_ratio/float(pid_current))*2, pid_current, 2*layer_x3, Nlayer
    

for pid in range(Np):
    coords[pid][2] += presents[pid][2]
max_coord = max([coord[2] for coord in coords])
for pid in range(Np):
    coords[pid][2] = max_coord - coords[pid][2]

make_submission(presents, coords)


'''
from collections import deque
left, bottom, width, height, color, edgecolor = [], [], [], [], [], []
stack = deque([tree.root])
while len(stack) > 0:
    node = stack.popleft()
    if node.left == None:
        assert node.right == None                        
        pass
    else:
        stack.append(node.left)
        stack.append(node.right)
            
    left.append(node.coord[0])
    bottom.append(node.coord[1])
    width.append(node.size[0])
    height.append(node.size[1])
    color.append((0.0, 0.0, 0.0, 0.0))
    edgecolor.append((1.0, 0.0, 0.0, 1.0))
'''

'''

print sum(packed)
'''

