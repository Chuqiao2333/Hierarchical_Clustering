
import helper_function   as hf
import numpy             as np

class FileClass(object):
    
    def __init__(self, data):
        super(FileClass, self).__init__()

        self.data  = data

    
    def alignment(self):

        self.data, center_mean, center_std = hf.alignment(self.data)
        print('Mean of the center = ', center_mean)
        print('Std of the center = ', center_std)

    def getMaxDiff(self):

        self.max_diff = np.max(np.max(self.data,axis = 0),axis = 0)

        return self.max_diff

    def getMeanDiff(self):

        self.mean_diff = np.mean(np.mean(self.data,axis = 0),axis = 0)

        return self.mean_diff

    def getRThetaPlot(self):

        return hf.get_r_theta_plot(self.log_std_map)

    def getMasked(self, in_r, out_r):

        self.data = hf.add_mask(self.data,in_r,out_r)

    def getADF(self):

        self.adf = hf.quickHAADF(self.data)

        return self.adf

    def getManifold(self, manifold_file_path = None,  n_neighbors=15):

        if manifold_file_path == None:

            self.manifold = hf.getMainifoldStructure(self.std_masked_data,n_neighbors=n_neighbors)
        else:
            self.manifold = np.load(manifold_file_path)

        return self.manifold

    def getLogStdMap(self):

        self.log_std_map = hf.log_std_map(self.data)

        return self.log_std_map

    def getStdMasked(self, threshold):

        sel = self.log_std_map > threshold
        self.std_masked_data = self.data[:,:, sel] 


class ClusterClass(object):
    
    def __init__(self, data, cluster_data, parent = None, sub_idx = -1):

        super(ClusterClass, self).__init__()

        self.data         = data
        if len(np.shape(cluster_data)) == 2:
            self.cluster_data = cluster_data
        elif len(np.shape(cluster_data)) == 3:
            self.cluster_data = np.reshape(cluster_data, (np.shape(cluster_data)[0]*np.shape(cluster_data)[1], np.shape(cluster_data)[2]))
        elif len(np.shape(cluster_data)) == 4:
            self.cluster_data = np.reshape(cluster_data, (np.shape(cluster_data)[0]*np.shape(cluster_data)[1], np.shape(cluster_data)[2]*np.shape(cluster_data)[3]))

        self.parent       = parent
        self.sub_idx      = sub_idx


    def choose_k_cluster(self, show_sse = True):

        if show_sse == True:
	        sse = hf.show_WSS_line(self.cluster_data)

	        self.k = int(input())

        	return sse
        else:
        	self.k = int(input())

    def getCluster(self):

        self.labels, self.centers = hf.one_round_clustering(self.k, self.cluster_data)

    def getRealSpaceMap(self):

        curr_node = self
        curr_label = self.labels.copy()
        while curr_node.parent != None:
            temp_label = curr_node.parent.labels.copy()
            sel = temp_label != curr_node.sub_idx
            temp_label[sel] = 0
            sel1 = temp_label == curr_node.sub_idx

            
            temp_label[sel1] = curr_label.copy()

            curr_node = curr_node.parent
            curr_label = temp_label.copy()

        self.real_space_map = np.reshape(curr_label, (np.shape(self.data)[0], np.shape(self.data)[1]))
        
        return self.real_space_map

    def getClusterCenters(self):

        self.center_diff = np.zeros((self.k, np.shape(self.data)[2], np.shape(self.data)[3] ))

        for i in range(1,  (self.k + 1)):
            sel = self.real_space_map == i

            self.center_diff[i-1] = np.mean(self.data[sel],axis = 0)

        return self.center_diff

    def getSubClusters(self):


        self.sub_clusters = []
        for i in range(1, self.k + 1):
            sel = self.labels == i
            sub_cluster_data = self.cluster_data[sel] 
            self.sub_clusters.append(ClusterClass(data = self.data, cluster_data = sub_cluster_data , parent = self, sub_idx = i))










