import os
import time
import numpy as np
import sqlite3
from src.graph2histogram import histogram_construct, histogram_view
from src.utils.loader import Loader
from src.utils.ploter import plot_clusters
from src.tools.graph import BipartiteGraph, UnipartiteGraph
from src.eaglemine_test import eaglemine


def get_subgraph(edgelist, label_point, feature):
    con = sqlite3.connect(":memory:")
    cur = con.cursor()

    #create edge table
    sql_str = '''CREATE TABLE EDGE
                    (id INTEGER PRIMARY KEY AUTOINCREMENT'''
    for i in range(len(edgelist[0])):
        sql_str += ", " + edgelist[0][i] + " " + edgelist[1][i]
    sql_str += ");"
    cur.execute(sql_str)

    #insert data into edge table
    col_ids_str = str(edgelist[0])
    col_ids_length = len(edgelist[0])
    sql_str = "INSERT INTO EDGE " + col_ids_str + " VALUES " + construct_sql_value_placeholder(col_ids_length)
    cur.executemany(sql_str, edgelist[2])

    #create subgraph table
    for label in sorted(label_point.keys()):
        sql_str = '''CREATE TABLE SUBGRAPH{}
                    (id INT PRIMARY KEY,
                        vid INT NOT NULL);'''.format(label)
        cur.execute(sql_str)
    
    #insert data into subgraph table
    for label in sorted(label_point.keys()):
        temp_vertex_list = [(i, label_point[label][i]) for i in range(len(label_point[label]))]
        sql_str = "INSERT INTO SUBGRAPH{} VALUES ".format(label) + construct_sql_value_placeholder(2)
        cur.executemany(sql_str, temp_vertex_list)
        del temp_vertex_list

    #create index for column vid
    for label in sorted(label_point.keys()):
        sql_str = "CREATE UNIQUE INDEX sub_vid{} on SUBGRAPH{} (vid);".format(label, label)
        cur.execute(sql_str)

    subgraph = []
    sql_str = "SELECT sub.id"
    if feature == "outd2hub":
        for label in sorted(label_point.keys()):
            temp_sql = '''CREATE TABLE NEWEDGE{}
                            (id INTEGER PRIMARY KEY AUTOINCREMENT'''.format(label)
            insert_temp_sql = "INSERT INTO NEWEDGE{} ({}".format(label, edgelist[0][0])

            for i in range(0, col_ids_length):
                if i != 0:
                    sql_str += ", edge." + edgelist[0][i]
                    insert_temp_sql += ", " + edgelist[0][i]
                temp_sql += ", " + edgelist[0][i] + " " + edgelist[1][i]
            temp_sql += ");"
            insert_temp_sql += ") VALUES " + construct_sql_value_placeholder(col_ids_length)
            
            #create index for table edge
            index_for_table_edge = "CREATE INDEX edge_uid on EDGE ({});".format(edgelist[0][0])
            cur.execute(index_for_table_edge)
            
            #get map data from compressed uid to oid
            sql_str += ''' FROM SUBGRAPH{} AS sub
                        LEFT JOIN EDGE AS edge
                        ON sub.vid = edge.{}'''.format(label, edgelist[0][0])
            cur.execute(sql_str)
            temp_sub = cur.fetchall()

            #delete old subgraph table
            sql_str = "DROP TABLE SUBGRAPH{};".format(label)
            cur.execute(sql_str)

            #create new edge table store new map from compressed uid to oid
            cur.execute(temp_sql)

            #insert data into new edge table
            cur.executemany(insert_temp_sql, temp_sub)
            del temp_sub

            #create oid table to compress oid
            temp_sql = '''CREATE TABLE OID{}
                        (id INT PRIMARY KEY,
                        vid INT NOT NULL);'''.format(label)
            cur.execute(temp_sql)

            #get oid data from new edge table
            sql_str = "SELECT DISTINCT " + edgelist[0][1] + " FROM NEWEDGE{};".format(label)
            cur.execute(sql_str)
            temp_sub = cur.fetchall()
            
            #insert data into oid table
            insert_temp_sql = "INSERT INTO OID{} VALUES ".format(label) + construct_sql_value_placeholder(2)
            temp_vertex_list = [(i, temp_sub[i][0]) for i in range(len(temp_sub))]
            del temp_sub
            cur.executemany(insert_temp_sql, temp_vertex_list)
            del temp_vertex_list

            #create index for table oid
            sql_str = "CREATE UNIQUE INDEX oid_vid{} on OID{} (vid);".format(label, label)
            cur.execute(sql_str)

            #create index for table newedge
            sql_str = "CREATE INDEX newedge_oid{} on NEWEDGE{} ({});".format(label, label, edgelist[0][1])
            cur.execute(sql_str)

            #get map data from compressed uid to compressed oid
            sql_str = "SELECT edge.{}, sub.id".format(edgelist[0][0])
            for i in range(2, len(edgelist[0])):
                sql_str += ", edge." + edgelist[0][i]
            sql_str += ''' FROM OID{} AS sub
                        LEFT JOIN NEWEDGE{} AS edge
                        ON sub.vid = edge.{}'''.format(label, label, edgelist[0][1])
            cur.execute(sql_str)
            temp_sub = cur.fetchall()

            #construct return value
            temp_edgelist = [edgelist[0], edgelist[1]]
            temp_edgelist.append(tuple(temp_sub))
            subgraph.append(tuple(temp_edgelist))
            
            del temp_sub

            #delete oid{} and newedge{} table
            sql_str = "DROP TABLE OID{};".format(label)
            cur.execute(sql_str)
            sql_str = "DROP TABLE NEWEDGE{};".format(label)
            cur.execute(sql_str)

    else:
        for label in sorted(label_point.keys()):
            temp_sql = '''CREATE TABLE NEWEDGE{}
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            {} {}'''.format(label, edgelist[0][1], edgelist[1][1])
            insert_temp_sql = "INSERT INTO NEWEDGE{} ({}".format(label, edgelist[0][1])

            for i in range(0, len(edgelist[0])):
                if i == 1: continue
                temp_sql += ", " + edgelist[0][i] + " " + edgelist[1][i]
                insert_temp_sql += ", " + edgelist[0][i]
                sql_str += ", edge." + edgelist[0][i]
            temp_sql += ");"
            insert_temp_sql += ") VALUES " + construct_sql_value_placeholder(col_ids_length)

            #create index for table edge
            index_for_table_edge = "CREATE INDEX edge_oid on EDGE ({});".format(edgelist[0][1])
            cur.execute(index_for_table_edge)
            
            #get map data from compressed uid to oid
            sql_str += ''' FROM SUBGRAPH{} AS sub
                        LEFT JOIN EDGE AS edge
                        ON sub.vid = edge.{}'''.format(label, edgelist[0][1])
            cur.execute(sql_str)
            temp_sub = cur.fetchall()

            #delete old subgraph table
            sql_str = "DROP TABLE SUBGRAPH{};".format(label)
            cur.execute(sql_str)

            #create new edge table store new map from compressed oid to uid
            cur.execute(temp_sql)

            #insert data into new edge table
            cur.executemany(insert_temp_sql, temp_sub)
            del temp_sub

            #create uid table to compress oid
            temp_sql = '''CREATE TABLE UID{}
                        (id INT PRIMARY KEY,
                        vid INT NOT NULL);'''.format(label)
            cur.execute(temp_sql)

            #get uid data from new edge table
            sql_str = "SELECT DISTINCT " + edgelist[0][0] + " FROM NEWEDGE{};".format(label)
            cur.execute(sql_str)
            temp_sub = cur.fetchall()
            
            #insert data into uid table
            insert_temp_sql = "INSERT INTO UID{} VALUES ".format(label) + construct_sql_value_placeholder(2)
            temp_vertex_list = [(i, temp_sub[i][0]) for i in range(len(temp_sub))]
            del temp_sub
            cur.executemany(insert_temp_sql, temp_vertex_list)
            del temp_vertex_list

            #create index for table uid
            sql_str = "CREATE UNIQUE INDEX uid_vid{} on UID{} (vid);".format(label, label)
            cur.execute(sql_str)

            #create index for table newedge
            sql_str = "CREATE INDEX newedge_uid{} on NEWEDGE{} ({});".format(label, label, edgelist[0][0])
            cur.execute(sql_str)

            #get map data from compressed uid to compressed oid
            sql_str = "SELECT sub.id, edge.{}".format(edgelist[0][0])
            for i in range(2, len(edgelist[0])):
                sql_str += ", edge." + edgelist[0][i]
            sql_str += ''' FROM UID{} AS sub
                        LEFT JOIN NEWEDGE{} AS edge
                        ON sub.vid = edge.{}'''.format(label, label, edgelist[0][0])
            cur.execute(sql_str)
            temp_sub = cur.fetchall()

            #construct return value
            temp_edgelist = [edgelist[0], edgelist[1]]
            temp_edgelist.append(temp_sub)
            subgraph.append(tuple(temp_edgelist))
            
            del temp_sub

            #delete oid{} and newedge{} table
            sql_str = "DROP TABLE UID{};".format(label)
            cur.execute(sql_str)
            sql_str = "DROP TABLE NEWEDGE{};".format(label)
            cur.execute(sql_str)

    #close db connection
    con.close()

    return subgraph

def map_label_to_point(point2pos_file, hpos2label_file):
    pos_point = {}
    with open("temp/point2pos.out", "r") as fp:
        for line in fp.readlines():
            if line.startswith('#'):  continue
            tok = line.strip().split(',')
            vid = int(tok[0])
            pos = tuple(map(int, tok[1:]))
            pos_point[pos] = vid
    
    label_pos = {}
    with open("outputData/hpos2label.out", "r") as fp:
        for line in fp.readlines():
            if line.startswith('#'):  continue
            tok = line.strip().split(',')
            label_id = int(tok[2])
            pos = tuple(map(int, tok[:2]))
            if label_id > 0:
                if label_id not in label_pos:
                    label_pos[label_id] = []
                label_pos[label_id].append(pos)
    
    label_point = {}
    for label in sorted(label_pos.keys()):
        for pos in label_pos[label]:
            if label not in label_point:
                label_point[label] = []
            label_point[label].append(pos_point[pos])

    return label_point

def construct_sql_value_placeholder(val_amount):
    if val_amount < 1:
        return None
    else: 
        value_placeholder = "(?"
        value_placeholder += ",?" * (val_amount - 1)
        value_placeholder += ")"
        return value_placeholder

# feature: ind2aut or outd2hub
def graph_to_cluster(edgelist, feature = "outd2hub", isBigraph = True):
    # bipartite graph
    loader = Loader()

    # load data
    data = []
    for edge in edgelist:
        elem = []
        for idx in (0, 1):
            elem.append(int(edge[idx]))
        data.append(np.array(elem))
    edgelist_data = np.array(data)

    print("load graph edgelist done!")
    print("bipartite graph feature extraction")
    bi_graph = BipartiteGraph()
    bi_graph.set_edgelist(edgelist_data)
    bi_graph.get_node_degree()
    bi_graph.get_hits_score()

    tempdir = "temp/"
    tempfile = {
        "feature": "outd2hub_feature",
        "histogram": "histogram.out",
        "pts2pos": "point2pos.out",
        "hpos2avgfeat": "hpos2avgfeat.out"
    }

    outdir = "outputData/"
    output = {
        "hpos2label": "hpos2label.out",
        "histogram_cluster": "histogram_cluster.png"
    }
    if feature == "outd2hub":
        tempfile["feature"] = "outd2hub_feature"
        with open(tempdir + tempfile["feature"], 'w') as ofp:
            ofp.writelines("# {},{}\n".format(len(bi_graph.src_outd), 0))
            for src in range(len(bi_graph.src_outd)):
                ofp.writelines("{},{}\n".format(bi_graph.src_outd[src], bi_graph.src_hub[src]))
    else:
        tempfile["feature"] = "ind2aut_feature"
        with open(tempdir + tempfile["feature"], 'w') as ofp:
            ofp.writelines("# {},{}\n".format(len(bi_graph.dest_ind), 0))
            for dest in range(len(bi_graph.dest_ind)):
                ofp.writelines("{},{}\n".format(bi_graph.dest_ind[dest], bi_graph.dest_auth[dest]))

    gfeat = np.loadtxt(tempdir + tempfile["feature"], float, comments='#', delimiter=',')
    histogram_construct(gfeat, int(1), tempdir + tempfile["histogram"], tempdir + tempfile["pts2pos"], tempdir + tempfile["hpos2avgfeat"], int(2))
    eaglemine(tempdir + tempfile["histogram"], tempdir + tempfile["pts2pos"], tempdir + tempfile["hpos2avgfeat"], outdir, int(1), int(2), int(2))
    eaglemine_cluster_view(outdir + output["hpos2label"], outdir + output["histogram_cluster"])

def size_relabels(labels):
    clsdic = {}
    for l in labels:
        if l not in clsdic:
            if l != -1: clsdic[l] = len(clsdic)
            else:       clsdic[l] = l
    rlbs = [clsdic[l] for l in labels]
    return np.array(rlbs)

def eaglemine_cluster_view(hpos2label_infn, outfn=None, outlier_label=-1):
    hcube_label = np.loadtxt(hpos2label_infn, int, delimiter=',')
    outs_index = hcube_label[:, -1] == outlier_label
    outs = hcube_label[outs_index, :-1]
    others_cls = hcube_label[~outs_index]
    labels = size_relabels(others_cls[:, -1])
    cls_fig = plot_clusters(others_cls[:, :-1], [], labels, outliers=outs[::-1], ticks=False)
    if outfn is not None:
        cls_fig.savefig(outfn)

if __name__ == "__main__":
    tempdir = "../../../temp/"
    tempfile = {
        "feature": "outd2hub_feature",
        "histogram": "histogram.out",
        "pts2pos": "point2pos.out",
        "hpos2avgfeat": "hpos2avgfeat.out",
        "hpos2label": "hpos2label.out"
    }
    outdir = "../../../outputData/"
    eaglemine(tempdir + tempfile["histogram"], tempdir + tempfile["pts2pos"], tempdir + tempfile["hpos2avgfeat"], outdir, int(1), int(2), int(2))