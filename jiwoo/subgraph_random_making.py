import json
import random
import pickle
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

entity_matching={}
with open('/SSL_NAS/jiwoo2nd/subgraph/wikidata5m/wikidata5m/wikidata5m_alias (1)/wikidata5m_entity.txt', 'r', encoding='utf-8') as file:
    for line in file:
        x=line.strip().split('\t')
        entity_matching[x[0]]=x[1]

relation_matching={}
with open('/SSL_NAS/jiwoo2nd/subgraph/wikidata5m/wikidata5m/wikidata5m_alias (1)/wikidata5m_relation.txt', 'r', encoding='utf-8') as file:
    for line in file:
        x=line.strip().split('\t')
        relation_matching[x[0]]=x[1]


graph_data=[]
with open('/SSL_NAS/jiwoo2nd/subgraph/wikidata5m/wikidata5m/wikidata5m_all_triplet (1).txt/wikidata5m_all_triplet.txt', 'r', encoding='utf-8') as file:
    for line in file:
        x=line.strip().split('\t')
        graph_data.append(x)



subgraph_list=[]
adj_list=[]
flag=False #그래프 추출을처음부터 다시 해야할 때 True
graph_count=0 #추출한 그래프 개수
for i in tqdm(range(1000000)):
    tuple_idx = 0  # graph_data의 데이터 돌 것!

    a = []
    x=random.choice(list(entity_matching.keys()))
    a.append(x)
    adj=[]
    for k in range(10):
        b=[]
        for tmp in range(len(graph_data)):
            if graph_data[tuple_idx][0] in a and graph_data[tuple_idx][2] not in a:
                if graph_data[tuple_idx][2] not in b:
                    b.append(graph_data[tuple_idx][2])
            if graph_data[tuple_idx][2] in a and graph_data[tuple_idx][0] not in a:
                if graph_data[tuple_idx][0] not in b:
                    b.append(graph_data[tuple_idx][0])
            tuple_idx+=1009 #len(graph_data)와 서로소인 소수 1009 =>속도 급감
            if tuple_idx>=len(graph_data):
                tuple_idx-=len(graph_data)
            if len(b)>2:
                break
        if len(b)==0:
            flag=True
            break
        c=random.choice(b)
        a.append(c)
    if flag==True:
        flag=False
        continue
    a.sort()
    if a in subgraph_list:
        continue

    for n in range(len(graph_data)): #여기 n을 1009씩 더하는거로 바꿔야하나?
        if graph_data[n][0] in a and graph_data[n][2] in a:
            adj.append(graph_data[n])
        if len(adj) == 10:
            break

    graph_count+=1
    subgraph_list.append(a)
    adj_list.append(adj)
    if i%100000==0:
        print(i,graph_count,flush=True)
    if graph_count==3:
        break

print(adj_list)
exit()

with open("subgraph_list_entity_wiki.json", "w") as f:
    json.dump(subgraph_list, f)
with open("subgraph_list_adj_wiki.json", "w") as f:
    json.dump(adj_list, f)

string_representation=[]
for i in range(len(adj_list)):
    x=adj_list[i]
    for k in range(len(x)):
        x[k]=entity_matching[x[k][0]]+" | "+relation_matching[x[k][1]]+" | "+entity_matching[x[k][2]]
    string_of_graph=" '"
    for k in range(len(x)):
        if k==0:
            string_of_graph = string_of_graph + x[k] + "'"
        else:
            string_of_graph = string_of_graph + ", '"+x[k]+"'"
    string_representation.append(string_of_graph)

with open("subgraph_list_string_wiki.json", "w") as f:
    json.dump(string_representation, f)


print("done")
