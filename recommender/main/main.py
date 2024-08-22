import os
import gensim
import jieba.analyse
import langid
import nltk
import numpy as np
import pandas as pd
from gensim import models, similarities
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from recommender.Requester.ReposRequester import ReposRequester
from recommender.Requester.UserRequester import UserRequester


class Main:
    tfidf = models.TfidfModel
    dictionary = gensim.corpora.Dictionary
    index = similarities.SparseMatrixSimilarity
    corpus_list = []
    train = []
    train_item_id = []
    train_item_assignees = []
    idlist = []
    assignee_list = []
    train_contributors = []
    contributors = []
    similarity = []
    user_capacity = []
    user_read_ability = []
    user_list = []
    user_info_list = []

    def __init__(self):
        self.user_capacity = []
        self.user_read_ability = []
        self.similarity = []
        self.corpus_list = []
        self.train = []
        self.user_list = []
        self.train_item_id = []
        self.train_item_assignees = []
        self.idlist = []
        self.assignee_list = []
        self.train_contributors = []
        self.contributors = []
        self.load_tfidf()
        self.user_info_list = []

    @staticmethod
    def get_stopwords():
        with open('files/baidu_stopwords.txt') as f:
            stop_words = f.read()
        return stop_words.replace('\\n', "")

    # 加载tfidf
    def load_tfidf(self):
        self.tfidf = models.TfidfModel.load('model/data.tfidf')
        self.dictionary = gensim.corpora.Dictionary.load('files/词典.dic')
        self.index = similarities.SparseMatrixSimilarity.load('files/train.index')
        # with open("../files/词频.txt", "r", encoding="utf-8") as f:
        #     corpus = f.read()
        #     list1 = corpus.split('], [')
        #     for each in list1:
        #         each = each.replace('[[', '').replace(']]', '')
        #         # print(each)
        #         each_list = each.split('), (')
        #         each_list1 = []
        #         for item in each_list:
        #             # print(item)
        #             item = item.replace('(', '').replace(')', '').replace(' ', '')
        #             item = eval(item)  # (1, 2)
        #             # print(item)
        #             each_list1.append(item)  # [(), (), ……, ()]
        #         # print(each_list1)
        #         self.corpus_list.append(each_list1)
        for text in pd.read_csv("files/trainId.csv", encoding="utf-8").iterrows():
            repo_num = text[1][1].replace("]", "").replace("\'", "").replace("\"", "").split("[")
            self.train_item_id.append(repo_num[0].replace(",", "").replace("\'", ""))
            self.train_item_assignees.append(repo_num[1].split(", "))
            self.train_contributors.append(repo_num[2].split(", "))

    # 获取最相似的问题报告
    def get_similar_issue(self, text, language):
        stop_words = self.get_stopwords()
        lemma = WordNetLemmatizer()
        text = text.replace("\\n", "").replace("\\r", "")
        lan = langid.classify(text)
        new_text = []
        if lan[0] == 'zh':
            text = jieba.lcut(text, cut_all=True)
        else:
            text = nltk.word_tokenize(text)
        for w in text:
            w.replace("-", " ").replace("__", "").replace("\\", " ").replace("/", " ").split(" ")
            stop = set(stopwords.words('english'))
            if type(w) is str:
                if (w not in stop_words) & (w not in stop):
                    new_text.append(lemma.lemmatize(w))
            else:
                for i in w:
                    if (i not in stop_words) & (i not in stop):
                        new_text.append(lemma.lemmatize(w))
        # new_text = jieba.analyse.extract_tags(str(new_text), topK=20, withWeight=False, allowPOS=())
        vec_new = self.dictionary.doc2bow(document=new_text)
        vec_new_tfidf = self.tfidf[vec_new]
        self.index.num_best = 100
        sim = self.index.get_similarities(query=vec_new_tfidf)
        related_doc_indices = sim.argsort()[:-11:-1]
        for i in related_doc_indices:
            self.idlist.append(self.train_item_id[i])
            self.assignee_list.append(self.train_item_assignees[i])
            self.contributors.append(self.train_contributors[i])
            self.similarity.append(sim[i] * 100)
        self.get_user(language)

    # 获取user_info 文件里所有用户的影响因素并保存在user_ability和read_ability文件中
    def get_user_items(self, language: str):
        user_info = pd.read_csv("../files/user_info.csv", encoding="utf-8").drop_duplicates()
        repos = []
        r = ReposRequester()
        for row in user_info.iterrows():
            # for ind, d in enumerate(self.user_list):
            #     if row[1][2] == d["login"]:
            repos.append({"login": row[1][2],
                          "repo": row[1][7].replace("[", "").replace("]", "").replace("\'", "").split(", ")})
        for repo in repos:
            username = repo["login"]
            capacity = {"repo_num": 0, "language": {}, "code": 0, "stargazers_count": 0, "commit_num": 0}
            read = r.get_fork_commit(username)
            for reponame in repo["repo"]:
                detail = r.get_repo_details(username, reponame)
                capacity["repo_num"] += 1
                capacity["code"] += detail["code"]
                capacity["stargazers_count"] += detail["stargazers_count"]
                capacity["commit_num"] += detail["commit_num"]
                keys = list(detail["language"].keys())
                key = list(capacity["language"].keys())
                if language in keys:
                    value = detail["language"][language]
                    if {k: v for k, v in capacity["language"].items() if (k == language)}:
                        capacity["language"][language] += value
                    else:
                        capacity["language"].update({language: value})
                else:
                    if language in key:
                        capacity['language'][language] += 0
                    else:
                        capacity["language"].update({language: 0})
                # print(capacity)
            dic = {'login': repo['login'], 'reponum': capacity["repo_num"], 'code': capacity["code"],
                   'star': capacity["stargazers_count"], 'commit': capacity["commit_num"],
                   language: capacity["language"][language]}
            df = pd.DataFrame(dic, index=[0])
            file_path = '../files/user_ability.csv'
            if os.path.exists(file_path):
                df.to_csv(file_path, mode="a", header=False)
            else:
                df.to_csv(file_path, mode="w")

            dic2 = {'login': repo['login'], 'fork_num': read['fork_num'], 'commit': read["commit"],
                    'star': read["star"]}
            df2 = pd.DataFrame(dic2, index=[0])
            file_path2 = '../files/read_ability.csv'
            if os.path.exists(file_path):
                df2.to_csv(file_path2, mode="a", header=False)
            else:
                df2.to_csv(file_path2, mode="w")

    # 计算分数
    def calculator(self, language):
        ability = pd.read_csv('../files/user_ability.csv', encoding='utf-8')
        read = pd.read_csv('../files/read_ability.csv', encoding='utf-8')
        items_1 = []
        items_2 = []
        user = []
        login = []
        w2 = [0.10267543859649123, 0.08127192982456138, 0.07140350877192983, 0.20710526315789476, 0.28754385964912277]
        w3 = [0.06296296296296296, 0.14722222222222223, 0.03981481481481481]
        if language == 'Objective-C':
            for r1 in ability.iterrows():
                items_1.append(
                    [r1[1][2] * w2[0], r1[1][3] * w2[1], r1[1][4] * w2[2], r1[1][5] * w2[3], r1[1][6] * w2[4]])
                login.append(r1[1][1])
        else:
            if language == 'Clojure':
                x = 7
            elif language == 'Java':
                x = 8
            elif language == 'Go':
                x = 9
            elif language == 'JavaScript':
                x = 10
            else:
                x = 11
            for r1 in ability.iterrows():
                items_1.append(
                    [r1[1][2] * w2[0], r1[1][3] * w2[1], r1[1][4] * w2[2], r1[1][5] * w2[3], r1[1][x] * w2[4]])
                login.append(r1[1][1])
        for r2 in read.iterrows():
            items_2.append([r2[1][2] * w3[0], r2[1][3] * w3[1], r2[1][4] * w3[2]])
        for index in range(len(items_1)):
            user.append(items_1[index] + items_2[index])
        x = np.array(user)
        Xmax, Xmin = x.max(axis=0), x.min(axis=0)
        x_minmax = ((x - Xmin) / (Xmax - Xmin))
        for i in range(len(x_minmax)):
            score = x_minmax[i][0] + x_minmax[i][1] + x_minmax[i][2] + \
                    x_minmax[i][3] + x_minmax[i][4] + \
                    x_minmax[i][5] + x_minmax[i][6] + x_minmax[i][7]
            score = score * 100
            if 0 <= score < 20:
                level = 1
            elif 20 <= score < 40:
                level = 2
            elif 40 <= score < 60:
                level = 3
            elif 60 <= score < 80:
                level = 4
            else:
                level = 5
            scoredict = {'login': login[i], 'score': score, 'level': level}
            return scoredict
            # print(scoredict)
            # df = pd.DataFrame(scoredict, index=[0])
            # file_path = '../files/user_score_t.csv'
            # if os.path.exists(file_path):
            #     df.to_csv(file_path, mode="a", header=False)
            # else:
            #     df.to_csv(file_path, mode="w")

    # 最终推荐
    def get_user(self, language):
        users = []
        for i in range(len(self.assignee_list)):
            if self.assignee_list[i] == ['']:
                users.append({"loginList": self.contributors[i], "issue": self.idlist[i], "type": 2,
                              "similarity": self.similarity[i]})
            else:
                users.append({"loginList": self.assignee_list[i], "issue": self.idlist[i], "type": 1,
                              "similarity": self.similarity[i]})
            # break
        for index in range(len(users)):  # 获取每个issue关联的用户列表
            issue = users[index]["issue"]
            similarity = users[index]['similarity']
            if users[index]["type"] == 1:  # 有assignees直接加入userList
                self.user_list.append(
                    {"login": users[index]["loginList"], "issue": issue, "similarity": similarity, "type": 1})
            else:  # 没有则用contributors，打分加入
                con_list = []
                for login in users[index]['loginList']:
                    df = pd.read_csv("files/user_score.csv")
                    for r1 in df.iterrows():
                        if r1[1][1] == login.replace("'", ""):
                            if language == 'Objective-C':
                                x = 2
                            elif language == 'Clojure':
                                x = 4
                            elif language == 'Java':
                                x = 6
                            elif language == 'Go':
                                x = 8
                            elif language == 'JavaScript':
                                x = 10
                            else:
                                x = 12
                            score = r1[1][x]
                            level = r1[1][x + 1]
                            con_list.append({"login": login, "score": score, "level": level})
                        else:
                            continue
                con_list = sorted(con_list, key=lambda k: (k.get('score')))
                self.user_list.append({"login": con_list[:-6:-1], "issue": issue, "similarity": similarity, "type": 2})
        name_list = []
        for user in self.user_list:
            df = pd.read_csv('files/user_info.csv')
            if user['type'] == 2:
                for lo in user['login']:
                    if lo['login'] not in name_list:
                        for r3 in df.iterrows():
                            if r3[1][2] == lo['login']:
                                self.user_info_list.append(
                                    {'id': r3[1][1], 'name': r3[1][2], 'email': r3[1][5], 'location': r3[1][6],
                                     'homepage': 'https://github.com/' + r3[1][2],
                                     'resemblence': user['similarity'], 'level': lo['level']})
                                name_list.append(r3[1][2])
                    else:
                        continue

            else:
                for lo in user['login']:
                    if lo not in name_list:
                        for row in df.iterrows():
                            if row[1][2] == lo:
                                self.user_info_list.append(
                                    {'id': row[1][1], 'name': row[1][2], 'email': row[1][5], 'location': row[1][6],
                                     'homepage': 'https://github.com/' + row[1][2],
                                     'resemblence': user['similarity']})
                                name_list.append(row[1][2])


if __name__ == '__main__':
    m = Main()
    u = UserRequester()
    r = ReposRequester()
    m.get_similar_issue("Describe the bug Scatter plot with bubble size no longer renders bubbles for negative y axis "
                        "values. This appears to be a regression in 0.43, it worked in <0.43. To Reproduce Steps to "
                        "reproduce the behavior: Create a native query: select 1 x, 1 y, 20 size union all  select 2 "
                        "x, 10 y, 10 size union all  select 3 x, -9 y, 6 size union all  select 4 x, 100 y, "
                        "30 size union all  select 5 x, -20 y, 70 size Visualization: scatter, X axis: x, "
                        "Y axis: y At this point you will see points below 0 on the Y axis Set bubble size: size All "
                        "points below 0 on Y axis have disappeared Expected behavior I expect to see points <0 on Y "
                        "axis rendered the same as points with >=0 on Y axis",
                        "Clojure")
    # with open("../files/body.txt", encoding="utf-8") as f:
    # body = f.read()
    # m.get_similar_issue(body)      # 获得相似度排名
    print(m.similarity)
    print(m.idlist)
    # print(m.assignee_list)
    # print(m.contributors)
    # print(m.user_list)
    # m.user_list = [{'login': 'ranquild', 'score': 0}]
    # m.user_read_ability = [{'login': 'ranquild', 'r': {'file': 0.0, 'code': 0.0}}]
    # m.user_capacity = [{'login': 'ranquild',
    #                     'c': {'repo_num': 0.04, 'language': {'Objective-C': 0, 'java': 0.0}, 'code': 0.0796,
    #                           'stargazers_count': 0.0, 'commit_num': 1.0}}]
    # m.calculator("Java")
    # m.train_level()
