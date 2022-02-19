from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py
import numpy as np
from skimage.measure import profile_line
from skimage import transform

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Greys256
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
import scipy
import sys

from   sklearn.cluster import KMeans
from   sklearn.cluster import MiniBatchKMeans
from   sklearn.cluster import SpectralClustering
# from   yellowbrick.cluster import KElbowVisualizer
import umap
from mpl_toolkits.axes_grid1 import make_axes_locatable
# output_notebook()

def read_4D(fname, trim_meta = True):

    '''
    Read the 4D dataset as a numpy array from .raw , . mat, .npy file.
    Input:

    fname: the file path

    Return: 

    dp       : numpy array
    dp_shape : the shape of the data

    '''

    fname_end = fname.split('.')[-1]

    if fname_end == 'raw':
        with open(fname, 'rb') as file:
            dp = np.fromfile(file, np.float32)

        columns = 128    
        rows = 130
            
        sqpix = dp.size/columns/rows
        #Assuming square scan, i.e. same number of x and y scan points
        pix = int(sqpix**(0.5))
        
        dp = np.reshape(dp, (pix, pix, 130, 128), order = 'C')
        
        #Trim off the last two meta data rows if desired.  The meta data is for EMPAD debugging, and generally doesn't need to be kept.
        if trim_meta:
            dp = dp[:,:,0:128,:]

    ## Read 4D data from .mat file

    elif fname_end == 'mat':

        with h5py.File(fname, "r") as f:
            
            data_name = list(f.keys())[0]
            dp = np.array(list(f[data_name]))
    elif fname_end == 'npy':
        dp = np.load(fname)
    else:
        print('The Format is WRONG!! Only support .mat , .raw & .npy file !!') 


    sel = dp < 1
    dp[sel] = 1
    
    return dp


def get_r_theta_plot(diff):

    """
    
    Change a diffraction pattern from (x, y) coordinate to (r, theta)

    """
    x, y = np.shape(diff)
    center = (x//2, y//2)
    lines_list = []
    for i in range(x):
        point_list = [(0,i), (x,i),(i,0),(i,x)]
        for j, point in enumerate(point_list):
            temp_line = profile_line(diff, center, point, mode='reflect', linewidth=1)
            lines_list.append(temp_line)

    # Change the image to log image scale and bin the theta

    log_map = np.zeros((len(lines_list), x//2))
    for i in range(len(lines_list)):
        log_map[i] = lines_list[i][:x//2]

    Bin_map = np.zeros((len(lines_list)//10 + 1, x//2))

    for i in range(len(Bin_map)):
        if 10*i+10 < len(log_map):
            Bin_map[i] = np.mean(log_map[10*i:10*(i+1)],axis = 0)
        else:
            Bin_map[i] = np.mean(log_map[10*i:],axis = 0)

    # Change the log image scale from 5 - 15 for visualization 

    log_Bin_map = np.log(Bin_map.T+1)
    # sel = log_Bin_map<5
    # log_Bin_map[sel] = 5
    # sel = log_Bin_map>15
    # log_Bin_map[sel] = 15

    return log_Bin_map

def add_mask(dp, in_r, out_r ):

    x,y,kx,ky = np.shape(dp)
    mask = np.ones((kx,ky))

    center_kx, center_ky = kx // 2 - 0.5, ky // 2 - 0.5

    for i in range(kx):
        for j in range(ky):
            if (i - center_kx) ** 2 + (j - center_ky) ** 2 <= in_r ** 2:
                mask[i][j] = 0
            if (i - center_kx) ** 2 + (j - center_ky) ** 2 >= out_r ** 2:
                mask[i][j] = 0
    masked_dp = dp * mask

    return masked_dp

def quickHAADF(cbed):
    

    '''

    Get ADF image from the 4D dataset


    '''
    if len(np.shape(cbed)) == 2:

        x_x, kx_kx = np.shape(cbed)
        cbed = np.reshape(cbed, (int(np.sqrt(x_x)), int(np.sqrt(x_x)),int(np.sqrt(kx_kx)),int(np.sqrt(kx_kx))))

    cbed_size   = (np.shape(cbed)[2],np.shape(cbed)[3])
    dim1, dim2  = np.shape(cbed)[0], np.shape(cbed)[1]
    center_x    = int(cbed_size[0]/2)
    center_y    = int(cbed_size[1]/2)
    conv_factor = 1                  # convert mrad to pixels
    inner_disk  = int(cbed_size[0]/4)   # inner radius for HAADF 50 typically
    outer_disk  = int(cbed_size[0]/2)   # outer radius for HAADF 120 typically
    
    # ADF STEM
    # inner disk 50 mrad from Peter's thesis
    # outer disk 250 mrad
    
    inner_disk  /= conv_factor  # conversion factor is 1.991 mrad/pixels
    outer_disk  /= conv_factor
    
    pix_x, pix_y = np.arange(cbed_size[0]), np.arange(cbed_size[1])
    rx , ry      = np.meshgrid(pix_x, pix_y)
    rx          -= center_x
    ry          -= center_y
    rr           = np.sqrt(rx ** 2 + ry ** 2)
    dmask        = (rr > inner_disk) * (rr < outer_disk)
    
    
    im_adfSTEM   = np.zeros((dim1, dim2))
    
    for i in range(dim1):
        for j in range(dim2):
            
            diff = cbed[i,j,:,:]
            im_adfSTEM[i][j] = np.sum(diff * dmask)
            
            
    return im_adfSTEM

def alignment(cbed_data):

    '''

    Align the diffraction patterns through the Center of mass of the center beam
    '''

    x, y, kx, ky = np.shape(cbed_data)
    com_x, com_y = quickCOM(cbed_data) # need to add
    cbed_tran    = np.zeros((x, y, kx, ky))
    
    for i in range(x):
        for j in range(y):
            afine_tf = transform.AffineTransform(translation=(-kx//2+com_x[i,j], -ky//2+com_y[i,j]))
            cbed_tran[i,j,:,:] = transform.warp(cbed_data[i,j,:,:], inverse_map=afine_tf)
        sys.stdout.write('\r %d,%d' % (i, j) + ' '*10)
    com_x2, com_y2 = quickCOM(cbed_tran)
    std_com = (np.std(com_x2), np.std(com_y2))
    mean_com = (np.mean(com_x2), np.mean(com_y2))
    
    return cbed_tran, mean_com, std_com

def quickCOM(cbed_data):
    p = 6
    x, y, kx, ky = np.shape(cbed_data)
    center_x = kx//2 ; center_y = ky//2 
    disk = kx//6
    mask = spotmask(center_x,center_y, kx, disk)
    # ap2_x = np.zeros((x,y)); ap2_y = np.zeros((x,y))
    # masked_image = cbed_data * mask
    
    ap2_x, ap2_y = centroid2(cbed_data,x, y, kx, mask)
    # for i in range(x):
    #   for j in range(y):
    #       CoM = scipy.ndimage.center_of_mass(masked_image[i][j])
    #       ap2_x[i][j] = CoM[0]
    #       ap2_y[i][j] = CoM[1]

    
    return ap2_x, ap2_y

def spotmask(center_x,center_y, kx, disk):
    
    # conv_factor = 3
    innerDisk   = disk

#     rx, ry      = np.meshgrid(kx, kx)
#     rxx = rx - center_x; ryy = ry - center_y;
#     rr          = np.sqrt(rxx ** 2 + ryy ** 2)
#     mask        = (rr < innerDisk)

    mask = np.zeros((kx,kx))
    for i in range(kx):
        for j in range(kx):
            if (i - center_x) ** 2 + (j - center_y) ** 2 < innerDisk ** 2:
                mask[i][j] = 1

    return mask

def centroid2(fun, x, y, kx, mask):

    ap2_x = np.zeros((x,y)); ap2_y = np.zeros((x,y))
    ap_cent = np.zeros((kx, kx))
    rx, ry  = np.meshgrid(kx, kx)
    vx = np.arange(kx); vy = np.arange(kx)
    for i in range(x):
        for j in range(y):
            cbed = np.squeeze(fun[i,j, :, :] * mask)
            pnorm = np.sum(cbed)
#             print('pnorm = ', pnorm)
            ap2_x[i,j] = np.sum(vx * np.sum(cbed, axis = 0))/pnorm
            ap2_y[i,j] = np.sum(vy * np.sum(cbed, axis = 1))/pnorm
            
    return ap2_x, ap2_y

def getFFT(data):

    x, y, kx, ky = np.shape(data)
    fft_data = np.zeros(np.shape(data))
    for i in range(x):
        for j in range(y):
            fft_data[i][j] = np.log(np.abs(np.fft.fftshift(np.fft.fft2(np.log(data[i][j]+1)))) + 1)
    return fft_data

def getMainifoldStructure(dp, n_neighbors = 20, min_dist = 0.1, n_components = 3):


    if len(np.shape(dp)) == 4:

        x,y,kx,ky = np.shape(dp)
        dp_vec = np.reshape(dp, (x*y, kx*ky))

    elif len(np.shape(dp)) == 3:

        x,y,k = np.shape(dp)
        dp_vec = np.reshape(dp, (x*y, k))

    fit = umap.UMAP(
            n_neighbors  = n_neighbors,
            min_dist     = min_dist,
            n_components = n_components,
            )
    mainifold_structure  = fit.fit_transform(dp_vec)

    return mainifold_structure

def show_WSS_line(data, low_k = 1, high_k = 10):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(low_k ,high_k))
    visualizer.fit(data)  
    visualizer.show()# Fit the data to the visualizer


def one_round_clustering(n_clusters, manifold_data):

    if np.shape(manifold_data)[1] > 1000:
        manifold_clustering_result = MiniBatchKMeans(n_clusters = n_clusters).fit(manifold_data)
    else:
        manifold_clustering_result = KMeans(n_clusters = n_clusters).fit(manifold_data)

        
    labels = manifold_clustering_result.labels_ + 1

    return labels, manifold_clustering_result.cluster_centers_


def embeddable_image(data):
    img_data = 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((128, 128), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png',)
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def visualize2DMAP(vector, embedding, labels, num_clusters, data_shape, name = "data"):

    '''

    Visualize the whole 4D dataset through the ADF image. 
    In each scanning point, we can check the diffraction pattern there
    '''

    x,y,kx, ky = data_shape
    vec_image = np.log(np.reshape(vector, (len(vector), kx,ky))+1)
    sel = vec_image>15
    vec_image[sel] = 15
    sel = vec_image<5
    vec_image[sel] = 5

    # vec_image = 16 * (vec_image - np.min(vec_image)) / np.ptp(vec_image)
#     reducer = umap.UMAP(n_neighbors=15,
#         min_dist=0.1,
#         n_components=2)
#     reducer.fit(vector)
#     embedding = reducer.transform(vector)
    
    digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
    digits_df['digit'] = [str(x) for x in labels]
    digits_df['image'] = list(map(embeddable_image, vec_image))

    datasource = ColumnDataSource(digits_df)
    color_mapping = CategoricalColorMapper(factors=[str(x) for x in range(num_clusters)],
                                           palette=Greys256)

    plot_figure = figure(
        title='Real space and its diffraction patterns in each point of ' + ' ' + name,
        plot_width=800,
        plot_height=800,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: The 50px 50px 50px 50px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Feature:</span>
            <span style='font-size: 18px'>@digit</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='digit', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)


def getClusterCenters(data, k, labels):

    centers = np.zeros((k, np.shape(data)[1]))

    for i in range(k):
        sel = labels == i
        centers[i] = np.mean(data[sel],axis = 0)

    return centers




def getSSELine(manifold_data, kmin = 1, kmax = 10):

    sse = []
    for k in range(1, kmax+1):
        kmeans = MiniBatchKMeans(n_clusters = k).fit(manifold_data)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(manifold_data)
        curr_sse = 0
        
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(manifold_data)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.mean((manifold_data[i] - curr_center) ** 2 )


        sse.append(curr_sse)
    return sse

def show_WSS_line(data, low_k = 1, high_k = 10):
    
    sse = getSSELine(data, low_k, high_k)
    plt.plot(np.arange(low_k, high_k + 1), sse)
    plt.show()

    return sse


    # if np.shape(data)[1] > 100:
    #     model = MiniBatchKMeans()
    # else:
    #     model = KMeans()
    # visualizer = KElbowVisualizer(model, k=(1,10))
    # visualizer.fit(data) 
    # visualizer.show()

def log_std_map(data):

    return np.log(np.std(np.std(data,axis = 0),axis = 0)+1)


