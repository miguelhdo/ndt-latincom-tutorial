# Standard Python libraries
import glob
import os
import shutil
import ssl
from collections import defaultdict

# General data science libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import gdown

# Deep learning libraries
import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.loader import DataLoader

# set random seeds
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

########################################################################################################################
# Dataset utilities.
########################################################################################################################


def parse_deployment(fn):
    # Parse the deployment identifier.
    deployment = int(fn.split("/")[-1].split("_")[-1].split(".")[0])
    return deployment


def parse_node_code(node_code):
    # Split the node code string. Example node codes: AP_A, STA_A1.
    parts = node_code.split("_")

    # Extract:
    #     - node type : access point (AP), or station (STA).
    #     - node id : unique node identifier within the deployment.
    node_type = parts[0]
    node_id = parts[1]

    # Extract the wlan identifier {A, B, .., L}. Note that the wlan
    # identifier determines the parent of the node in the graph.
    node_wlan = node_id[0]

    # Extract the wlan 'address', i.e., each station gets a unique
    # (integer) identifier in its wlan.
    node_wlan_addr = 0
    if node_type == "STA":
        node_wlan_addr = int(node_id[1:])

    return node_type, node_wlan, node_wlan_addr


def convert_airtime(airtime):
    airtime = np.array(airtime)
    air = np.zeros(8)  # max 8 channels are used
    air[:airtime.shape[0]] = airtime
    return np.mean(air)


########################################################################################################################
# Node deployment input utilities.
########################################################################################################################

def read_nodes(fn):
    df = pd.read_csv(fn, sep=";")
    data = df.to_dict(orient="records")
    return data

########################################################################################################################
# edge deployment input utilities.
########################################################################################################################


def euclidean_distance(pos_a, pos_b):
    distance = np.linalg.norm(pos_a - pos_b)  # L2 norm
    return distance


########################################################################################################################
# Simulator output utilities.
########################################################################################################################


def read_list(fn):
    with open(fn, "r") as f:
        line = next(f)
        data = [float(x) for x in line.strip().split(",")]
    return data


def read_list_of_lists(fn):
    data = []
    with open(fn, "r") as f:
        line = next(f)
        tmp = line.strip().split(";")
        tmp = [x for x in tmp if x]
        for r in tmp:
            data.append([float(x) for x in r.split(',')])
    return data


def read_matrix(fn):
    data = []
    with open(fn, "r") as f:
        for line in f:
            line = line.strip().replace(';', '')
            row = [float(x) for x in line.split(',')]
            data.append(row)
    return data

########################################################################################################################
# Read the many many many custom files into a single data structure ...
########################################################################################################################


def read_dataset(path):
    # Dataset is organised per scenario.
    scenarios = {
        "train": [
            "sce1a",
            "sce1b",
            "sce1c",
            "sce2a",
            "sce2b",
            "sce2c"
        ],
        "test": [
            "test_sce1",
            "test_sce2",
            "test_sce3",
            "test_sce4",
        ]
    }

    # Dataset is stored per split in a dictionary, where the keys are (scenario, deployment) tuples.
    dataset = {
        'train': defaultdict(dict),
        'test': defaultdict(dict)
    }

    for split in ['train', 'test']:
        # Load input node files (deployments).
        nodes_path = os.path.join(path, split, 'input_node_files')
        nodes = []
        for scenario in scenarios[split]:
            nodes_files = glob.glob(os.path.join(nodes_path, scenario, "*.csv"))
            for fn in sorted(nodes_files):
                deployment = parse_deployment(fn)
                data = read_nodes(fn)
                dataset[split][(scenario, deployment)]['nodes'] = data
                dataset[split][(scenario, deployment)]['simulator'] = {}

        # Load simulator output files.
        simulations_path = os.path.join(path, split, 'output_simulator')
        for scenario in scenarios[split]:
            # Load airtime.
            for measurement in ['airtime']:
                measurement_files = glob.glob(
                    os.path.join(simulations_path, f"{scenario}_output", f"{measurement}_*.csv"))
                for fn in sorted(measurement_files):
                    deployment = parse_deployment(fn) - 1
                    data = read_list_of_lists(fn)
                    dataset[split][(scenario, deployment)]['simulator'][measurement] = data

            # Load RSSI, SINR, and throughput.
            for measurement in ['rssi', 'sinr', 'throughput']:
                measurement_files = glob.glob(
                    os.path.join(simulations_path, f"{scenario}_output", f"{measurement}_*.csv"))
                for fn in sorted(measurement_files):
                    deployment = parse_deployment(fn) - 1
                    data = read_list(fn)
                    if split == 'test' and measurement == 'throughput':
                        continue
                    dataset[split][(scenario, deployment)]['simulator'][measurement] = data

            # Load interference.
            for measurement in ['interference']:
                measurement_files = glob.glob(
                    os.path.join(simulations_path, f"{scenario}_output", f"{measurement}_*.csv"))
                for fn in sorted(measurement_files):
                    deployment = parse_deployment(fn) - 1
                    data = read_matrix(fn)
                    dataset[split][(scenario, deployment)]['simulator'][measurement] = data

    # Split the training dataset into a training and validation dataset. A fixed split has been created beforehand.
    train_split = pd.read_csv(os.path.join(path, 'train', 'train.csv'), header=None).values.tolist()
    valid_split = pd.read_csv(os.path.join(path, 'train', 'valid.csv'), header=None).values.tolist()

    train_split = set([tuple(row) for row in train_split])
    valid_split = set([tuple(row) for row in valid_split])

    train_dataset = {}
    valid_dataset = {}

    for (scenario, deployment), data in dataset['train'].items():
        if (scenario, deployment) in train_split:
            train_dataset[(scenario, deployment)] = data
        elif (scenario, deployment) in valid_split:
            valid_dataset[(scenario, deployment)] = data
        else:
            raise Exception(f'Scenario {scenario} and deployment {deployment} not found in splits.')

    dataset['train'] = train_dataset
    dataset['valid'] = valid_dataset

    return dataset

########################################################################################################################
# Put the data in a graph, without changing it ...
########################################################################################################################


def create_raw_graph(sample):
    # Mappings.
    wlan_to_node_id = {}

    # Nodes and edges.
    nodes = []
    edges = []

    # Nodes mask: indicated if a node is an AP (0) or a STA (1)
    node_mask = []

    # Node features, targets, etc.
    node_features = []
    node_targets = []
    node_ap = []

    # Edge features.
    edge_features = []

    # Station and access point features.
    sample_nodes = sample['nodes']
    sample_rssi = sample['simulator']['rssi']
    sample_sinr = sample['simulator']['sinr']
    sample_interference = sample['simulator']['interference']

    # Access point only features.
    sample_airtime = sample['simulator']['airtime']

    # Targets.
    if 'throughput' in sample['simulator']:
        # Targets (train)
        sample_throughput = sample['simulator']['throughput']
    else:
        # Dummy targets (test)
        sample_throughput = [-1 for _ in range(len(sample_nodes))]

    k = 0
    for node_id, (node, rssi, sinr, throughput) in enumerate(
            zip(sample_nodes, sample_rssi, sample_sinr, sample_throughput)):
        node_type, node_wlan, node_wlan_addr = parse_node_code(node["node_code"])

        # Nodes, features, and targets.
        nodes.append(node_id)
        node_targets.append(throughput)
        # Node features [node_type, x(m), y(m), primary_channel, min_channel_allowed, max_channel_allowed, airtime]
        features = [
            node['node_type'],
            node['x(m)'],
            node['y(m)'],
            node['primary_channel'],
            node['min_channel_allowed'],
            node['max_channel_allowed']
        ]

        # Edges between stations and access points.
        if node_type == "AP":
            # Register access point.
            ap_node_id = node_id
            wlan_to_node_id[node_wlan] = node_id
            node_mask.append(0)
            # airtime as node feature for APs
            airtime = sample_airtime[k]
            airtime = convert_airtime(airtime)
            features.append(airtime)
            k += 1

        if node_type == "STA":
            # Create an edge between the AP and STA.
            ap_node_id = wlan_to_node_id[node_wlan]
            edges.append((ap_node_id, node_id))
            node_mask.append(1)
            # Node features [node_type, x(m), y(m), primary_channel, min_channel_allowed, max_channel_allowed, airtime=0]
            features.append(0)
            # sanity check
            if np.isnan(sinr):
                sinr = 0
            # Edge Features AP - STA [rssi, sinr, interference=0, distance, bw]
            pos_ap = np.asarray([sample_nodes[ap_node_id]['x(m)'], sample_nodes[ap_node_id]['y(m)']])
            pos_sta = np.asarray([node['x(m)'], node['y(m)']])
            distance = euclidean_distance(pos_ap, pos_sta)
            # Bandwidth = 20 MHz per used channel
            bw = (node['max_channel_allowed'] - node['min_channel_allowed'] + 1) * 20
            edge_features.append([rssi, sinr, 0, distance, bw])

        # Store the node id of the AP associated with the STA.
        # Note: APs are associated with themselves.
        node_ap.append(ap_node_id)

        # Store node features
        node_features.append(features)

    # Create the (AP, AP) edges.
    ap_node_ids = [node_id for (node_id, mask) in zip(nodes, node_mask) if mask == 0]
    sample_interference = np.array(sample_interference)
    num_rows, num_cols = sample_interference.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if i == j:
                # No self loops.
                continue
            edges.append((ap_node_ids[i], ap_node_ids[j]))
            # Edge Features AP - AP [rssi=0, sinr=0, interference, distance=0, bw=0]
            edge_features.append([0, 0, sample_interference[i, j], 0, 0])

    # Merge all info.
    graph = {
        # Nodes and edges.
        "nodes": nodes,
        "edges": edges,
        # Features.
        "node_features": node_features,
        "edge_features": edge_features,
        # Node targets.
        "node_targets": node_targets,
        # Utilities: masks and associations.
        "node_mask": node_mask,
        "node_ap": node_ap,
    }

    return graph

########################################################################################################################
# Put the data in a graph that pytorch geometric understands
########################################################################################################################


def create_graph(graph):
    # Create an edge index.
    edges = []
    edges.extend((a, b) for (a, b) in graph['edges'])
    edges.extend((b, a) for (a, b) in graph['edges'])

    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_index = edge_index.t().contiguous()

    # Create edge attributes.
    edge_attr = []
    edge_attr.extend(graph['edge_features'])
    edge_attr.extend(graph['edge_features'])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create node features.
    x = torch.tensor(graph["node_features"], dtype=torch.float)

    # Create node targets.
    y = torch.tensor(graph["node_targets"], dtype=torch.float)

    # Create node target mask.
    y_mask = torch.tensor([bool(m) for m in graph["node_mask"]])

    # Node (station) to access point mappings for post-processing.
    node_ap = torch.tensor(graph["node_ap"], dtype=torch.long)

    # Node positions.
    pos = x[:, 1:3].clone()

    # Create Data object.
    data = Data(
        # Input graph.
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        # Output node targets.
        y=y,
        y_mask=y_mask,
        # Utilities.
        node_ap=node_ap,
        # Referencing
        scenario=graph['scenario'],
        deployment=graph['deployment'],
    )

    return data


def create_preprocessors(graphs):
    # Extract coordinates.
    x_coord, y_coord = [], []
    for g in graphs:
        x, y = zip(*[(d[1], d[2]) for d in g['node_features']])
        x_coord.extend(x)
        y_coord.extend(y)

    x_coord = np.array(x_coord).reshape(-1, 1)
    y_coord = np.array(y_coord).reshape(-1, 1)

    # Fit standard scalers.
    standard_scaler_x = StandardScaler()
    standard_scaler_x.fit(x_coord)

    standard_scaler_y = StandardScaler()
    standard_scaler_y.fit(y_coord)

    # Fit range scalers.
    range_scaler_x = MinMaxScaler()
    range_scaler_x.fit(x_coord)

    range_scaler_y = MinMaxScaler()
    range_scaler_y.fit(y_coord)

    # Extract channel configuration info.
    channel_info = []
    for g in graphs:
        c = [[d[3], d[4], d[5]] for d in g['node_features']]
        channel_info.extend(c)
    channel_info = np.array(channel_info)
    channel_info = pd.DataFrame(channel_info, columns=['primary_channel', 'min_channel_allowed', 'max_channel_allowed'])

    # Create a mapping from (primary, min, max) tuples to an integer id.
    channel_configs = {}
    for i, channel_config in enumerate(channel_info.drop_duplicates().values):
        channel_configs[tuple(channel_config)] = i

    # Fit a channel config (one-hot) encoder.
    channel_config_ids = np.array(list(channel_configs.values())).reshape(-1, 1)
    channel_config_encoder = OneHotEncoder(sparse_output=False)
    channel_config_encoder.fit(channel_config_ids)

    # Transform channel config ids.
    channel_config_encoder.transform(channel_config_ids)

    # Extract Airtime as last node feature
    airtime = []
    for g in graphs:
        a = [d[6] for d in g['node_features']]
        airtime.extend(a)
    airtime = np.array(airtime).reshape(-1, 1)

    # Fit standard scalers.
    standard_scaler_airtime = StandardScaler()
    standard_scaler_airtime.fit(airtime)

    # Fit range scalers.
    range_scaler_airtime = MinMaxScaler()
    range_scaler_airtime.fit(airtime)

    sinr, interference, rssi, distance, bw, = [], [], [], [], []
    for g in graphs:
        r, s, i, d, b = zip(*[(d[0], d[1], d[2], d[3], d[4]) for d in g['edge_features']])
        sinr.extend(s)
        interference.extend(i)
        rssi.extend(r)
        distance.extend(d)
        bw.extend(b)

    sinr = np.array(sinr).reshape(-1, 1)
    interference = np.array(interference).reshape(-1, 1)
    rssi = np.array(rssi).reshape(-1, 1)
    distance = np.array(distance).reshape(-1, 1)
    bw = np.array(bw).reshape(-1, 1)

    # Fit standard scalers.
    standard_scaler_sinr = StandardScaler()
    standard_scaler_sinr.fit(sinr)

    standard_scaler_interference = StandardScaler()
    standard_scaler_interference.fit(interference)

    standard_scaler_rssi = StandardScaler()
    standard_scaler_rssi.fit(rssi)

    standard_scaler_distance = StandardScaler()
    standard_scaler_distance.fit(distance)

    standard_scaler_bw = StandardScaler()
    standard_scaler_bw.fit(bw)

    # Fit range scalers.
    range_scaler_sinr = MinMaxScaler()
    range_scaler_sinr.fit(sinr)

    range_scaler_interference = MinMaxScaler()
    range_scaler_interference.fit(interference)

    range_scaler_rssi = MinMaxScaler()
    range_scaler_rssi.fit(rssi)

    range_scaler_distance = MinMaxScaler()
    range_scaler_distance.fit(distance)

    range_scaler_bw = MinMaxScaler()
    range_scaler_bw.fit(bw)

    preprocessors = {
        'x': {
            'standard': standard_scaler_x,
            'range': range_scaler_x,
        },
        'y': {
            'standard': standard_scaler_y,
            'range': range_scaler_y,
        },
        'channel_info': {
            'categorical': channel_configs,
            'one_hot': channel_config_encoder,
        },
        'sinr': {
            'standard': standard_scaler_sinr,
            'range': range_scaler_sinr,
        },
        'airtime': {
            'standard': standard_scaler_airtime,
            'range': range_scaler_airtime,
        },
        'interference': {
            'standard': standard_scaler_interference,
            'range': range_scaler_interference,
        },
        'rssi': {
            'standard': standard_scaler_rssi,
            'range': range_scaler_rssi,
        },
        'distance': {
            'standard': standard_scaler_distance,
            'range': range_scaler_distance,
        },
        'bw': {
            'standard': standard_scaler_bw,
            'range': range_scaler_bw,
        }

    }

    return preprocessors

########################################################################################################################
# Prepare the data for learning!
########################################################################################################################


def preprocess_graph(graph, preprocessors):
    node_features = graph['node_features']
    edge_features = graph['edge_features']

    # Pre-process node type.
    node_type = np.array([d[0] for d in node_features]).reshape(-1, 1)

    # Pre-process coordinates.
    x_coord, y_coord = zip(*[(d[1], d[2]) for d in node_features])
    x_coord = np.array(x_coord).reshape(-1, 1)
    y_coord = np.array(y_coord).reshape(-1, 1)

    x_coord = preprocessors['x']['range'].transform(x_coord)
    y_coord = preprocessors['y']['range'].transform(y_coord)

    # Pre-process channel configuration: categorical encoding (step 1).
    channel_configs = [(d[3], d[4], d[5]) for d in graph['node_features']]
    channel_configs = [preprocessors['channel_info']['categorical'][c] for c in channel_configs]

    # Pre-process channel configuration: one-hot encoding (step 2).
    channel_configs = np.array(channel_configs).reshape(-1, 1)
    channel_configs = preprocessors['channel_info']['one_hot'].transform(channel_configs)

    # Pre-process airtime
    airtime = [d[6] for d in node_features]
    airtime = np.array(airtime).reshape(-1, 1)

    airtime = preprocessors['airtime']['range'].transform(airtime)

    # Pre-process everything else
    rssi, sinr, interference, distance, bw, = zip(
        *[(d[0], d[1], d[2], d[3], d[4]) for d in edge_features])

    # Pre-process sinr
    sinr = np.array(sinr).reshape(-1, 1)
    sinr = preprocessors['sinr']['range'].transform(sinr)

    # Pre-process interference
    interference = np.array(interference).reshape(-1, 1)
    interference = preprocessors['interference']['range'].transform(interference)

    # Pre-process rssi
    rssi = np.array(rssi).reshape(-1, 1)
    rssi = preprocessors['rssi']['range'].transform(rssi)

    # Pre-process distance
    distance = np.array(distance).reshape(-1, 1)
    distance = preprocessors['distance']['range'].transform(distance)

    # Pre-process bandwidth
    bw = np.array(bw).reshape(-1, 1)
    bw = preprocessors['bw']['range'].transform(bw)

    # Create preprocessed feature vector.
    node_features_processed = [
        node_type,
        x_coord,
        y_coord,
        channel_configs,
        airtime,
    ]
    node_features_processed = np.concatenate(node_features_processed, axis=1)

    # Pre-process edge features
    edge_features_processed = [
        rssi,
        sinr,
        interference,
        distance,
        bw
    ]
    edge_features_processed = np.concatenate(edge_features_processed, axis=1)

    # Update graph.
    graph['node_features'] = node_features_processed
    graph['edge_features'] = edge_features_processed

    return graph

########################################################################################################################
# Pytorch Geometric in-memory dataset implementation.
########################################################################################################################


class NDTDataset(InMemoryDataset):
    # we create an InMemoryDataset since the whole dataset fits into CPU memory.
    # get a direct link to download a file from Google Drive by replacing the shareable url with
    # https://drive.google.com/uc?export=download&id=DRIVE_FILE_ID and the corresponding file id.
    dataset_url = "https://drive.google.com/file/d/1mrsZ-HlkZCAVT8ajo_wWcCREl5D2gIJ3/view?usp=sharing"
    #dataset_url = "https://drive.google.com/uc?export=download&id=14rD7TjcSLw6Qxouk2rdgW72Op-byfuU_"

    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super(NDTDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if split == 'train':
            data_path = self.processed_paths[0]
        elif split == 'valid':
            data_path = self.processed_paths[1]
        elif split == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("Invalid split")
        self.data, self.slices = torch.load(data_path)

    @property
    def raw_file_names(self):
        return ['train', 'test']

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt','test.pt']

    def makedirs(self, path):
        try:
            os.makedirs(os.path.expanduser(os.path.normpath(path)))
        except OSError as e:
            if e.errno != errno.EEXIST and os.path.isdir(path):
                raise e
    def download_url(self, url, folder, log=True, retries=3, delay=5):
        filename = url.rpartition('/')[2]
        path = os.path.join(folder, filename)
    
        print(path)
    
        if os.path.exists(path):  # pragma: no cover
            if log:
                print('Using existing file', filename)
            return path
    
        if log:
            print('Downloading', url)
    
        self.makedirs(folder)
        context = ssl._create_unverified_context()
    
        # If Google Drive link, use gdown instead
        if "drive.google.com" in url:
            file_id = url.split('/')[-2]  # Extract file ID from URL
            gdown_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(gdown_url, path, quiet=not log)
            return path
    
        # Standard download for non-Google Drive URLs
        for attempt in range(retries):
            try:
                data = urllib.request.urlopen(url, context=context)
                with open(path, 'wb') as f:
                    f.write(data.read())
                return path
            except Exception as e:
                print(f"Download failed: {e}. Retrying {attempt + 1}/{retries}")
                time.sleep(delay)
        raise Exception("Download failed after multiple attempts")


    def download(self):
        # Prepare raw data directory.
        shutil.rmtree(self.raw_dir)

        # Download and extract the dataset.
        path = self.download_url(self.dataset_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        datasets = read_dataset(self.raw_dir)

        preprocessors = None
        for split in ['train', 'valid', 'test']:
            print(f"Processing {split} split.")
            # Read data for each split into a huge `Data` list.
            graphs = []
            for (scenario, deployment), sample in datasets[split].items():
                graph = create_raw_graph(sample)
                graph['scenario'] = scenario
                graph['deployment'] = deployment
                graphs.append(graph)

            if split == 'train':
                # Analyse data and fit preprocessors (e.g., scalers, encoders).
                preprocessors = create_preprocessors(graphs)

            # Pre-process graph (feature scaling and encoding).
            data_list = []
            for graph in graphs:
                graph = preprocess_graph(graph, preprocessors)
                graph = create_graph(graph)
                data_list.append(graph)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)

            if split == 'train':
                output_path = self.processed_paths[0]
            elif split == 'valid':
                output_path = self.processed_paths[1]
            elif split == 'test':
                output_path = self.processed_paths[2]
            else:
                raise Exception("Invalid split")

            torch.save((data, slices), output_path)


# ########################################################################################################################
# # Dataset.
# ########################################################################################################################
#
# root = './datasets/NDTDataset/gnn/'
# # remove the following line if you don't want to download and process the dataset
# shutil.rmtree(root)
#
# # Load training dataset.
# dataset_train = NDTDataset(root, split='train')
# dataset_valid = NDTDataset(root, split='valid')
# dataset_test = NDTDataset(root, split='test')