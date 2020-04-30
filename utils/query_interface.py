import json
import re

from SPARQLWrapper import SPARQLWrapper, JSON

# DBpedia_endpoint = "https://dbpedia.org/sparql"
# DBpedia_endpoint = "http://10.201.180.179:8890/sparql"
DBpedia_endpoint = "http://10.201.180.179:3030/dbpedia/sparql"


def DBpedia_query(_query):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(DBpedia_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()
    results = parse_query_results(response, _query)
    return results

def parse_query_results(response, query):
    if 'ASK' in query:  # ask
        result = response['boolean']
    elif 'COUNT' in query:  # count
        result = int(response['results']['bindings'][0]['callret-0']['value'])
    else:
        result = []
        for res in response['results']['bindings']:
            res = {k: v["value"] for k, v in res.items()}
            result.append(res)
    return result

def get_true_answer(data):
    response = DBpedia_query(data['sparql_query'])
    if data['classx'] == '4':  # ask
        res = response['boolean']
    elif data['classx'] in ['0', '1']:  # select
        uri_res = [
            x['uri']['value'] for x in response['results']['bindings']
        ]
        res = []
        for uri in uri_res:
            uri = str(uri).split('/')
            res.append(uri[-1].lower())
        # print(res)
    elif data['classx'] in ['2', '3']:  # count
        res = int(response['results']['bindings'][0]['callret-0']['value'])
    else:  # other
        res = None
    return res


def ans_rate():
    # dataset = json.load(open('../data/test-data.json', 'r', encoding='UTF-8')) + \
    #           json.load(open('../data/train-data.json', 'r', encoding='UTF-8'))
    dataset = json.load(open('../data/FullyAnnotated_LCQuAD5000.json', 'r', encoding='UTF-8'))
    # doubt = []
    num = 0
    no_ans = 0
    for node in dataset:
        response = DBpedia_query(node['sparql_query'])
        if 'ASK' in node['sparql_query']:  # ask
            ans = response['boolean']
            if type(ans) == bool:
                num += 1
                # if not node['is_get']:
                #     doubt.append(node['origin_data'])
            if not ans:
                no_ans += 1
        elif 'COUNT' in node['sparql_query']:  # count
            ans = int(response['results']['bindings'][0]['callret-0']['value'])
            if ans > 0:
                num += 1
                # if not node['is_get']:
                #     doubt.append(node['origin_data'])
        elif 'SELECT' in node['sparql_query']:  # select
            uri_res = [
                x['uri']['value'] for x in response['results']['bindings']
            ]
            ans = []
            for uri in uri_res:
                uri = str(uri).split('/')
                ans.append(uri[-1].lower())
            if ans:
                num += 1
                # if not node['is_get']:
                #     doubt.append(node['origin_data'])
            # else:
            #     print(ans)
    # json.dump(doubt, open('doubt.json', 'w+'), indent=4)
    # print('size of which have ans but no pred chains ', len(doubt))
    # 4998/5000 0
    print('ans : %d / %d, ' % (num, len(dataset)), ' ASK No num :', no_ans)

    # dataset = json.load(open('../data/data.json', 'r'))
    # doubt = []
    # num = 0
    # c = [0] * 5
    # no_ans = 0
    # for node in dataset:
    #     ans = get_true_answer(node['origin_data'])
    #     c[int(node['origin_data']['classx'])] += 1
    #     if node['origin_data']['classx'] == '4':  # ask
    #         if type(ans) == bool:
    #             num += 1
    #             if not node['is_get']:
    #                 doubt.append(node['origin_data'])
    #         if not ans:
    #             no_ans += 1
    #     elif node['origin_data']['classx'] in ['0', '1']:  # select
    #         if ans:
    #             num += 1
    #             if not node['is_get']:
    #                 doubt.append(node['origin_data'])
    #         else:
    #             print(ans)
    #     elif node['origin_data']['classx'] in ['2', '3']:  # count
    #         if ans > 0:
    #             num += 1
    #             if not node['is_get']:
    #                 doubt.append(node['origin_data'])
    # json.dump(doubt, open('doubt.json', 'w+'), indent=4)
    # print('size of which have ans but no pred chains ', len(doubt))
    # # 4998/5000 0
    # print('ans : %d / %d, ' % (num, len(dataset)), 'classx : ', c, ' ASK No num :', no_ans)


if __name__ == '__main__':
    # from datetime import datetime
    #
    # start_time = datetime.now()
    # ans_rate()
    # print('used time ', datetime.now() - start_time)

    query = " SELECT DISTINCT ?uri WHERE { ?x rdf:type ?uri } "
    results = DBpedia_query(query)

    fout = open("../data/type.txt", "w", encoding='utf-8')
    for res in results:
        if res["uri"].find("http://dbpedia.org/ontology/") == -1:
            continue
        fout.write(res["uri"] + "\n")
