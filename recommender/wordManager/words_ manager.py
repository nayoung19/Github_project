import gensim.models
import nltk
import jieba
import jieba.analyse
import pandas as pd
import langid
import os
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from recommender.Entity.Issue_Entity import IssueEntity
from recommender.Entity.Repo import Repo
from recommender.Entity.User import User


class WordsManager:
    repo_name_list = []
    issue_num_list = []
    user_id_list = []
    user_login_list = []
    user_closed_id = []
    user_closed_login = []
    issue_title_list = []
    issue_body_list = []
    issue_assignees_list = []
    issue_label_list = []
    repo_language_list = []
    repo_topic_list = []
    repo_description_list = []
    repo_contributors_list = []
    repo_readme_list = []
    issue_comment_list = []
    df_issue = {}
    df_comment = {}
    index_issue = ["repo_name", "num", "userId", "userLogin", "closeId", "closeLogin", "title", "body",
                   "assignees", "labels", "language", "topics", "description", "contributors", "readme"]
    index_comment = ["num", "userId", "userLogin", "body", "author_association"]

    def __init__(self):
        self.clear_list()
        self.result_path = os.path.dirname(os.getcwd()) + os.sep + "files" + os.sep

    def clear_list(self):
        self.repo_name_list = []
        self.issue_num_list = []
        self.user_id_list = []
        self.user_login_list = []
        self.user_closed_id = []
        self.user_closed_login = []
        self.issue_title_list = []
        self.issue_body_list = []
        self.issue_assignees_list = []
        self.issue_label_list = []
        self.repo_language_list = []
        self.repo_topic_list = []
        self.repo_description_list = []
        self.repo_contributors_list = []
        self.issue_comment_list = []

    def set_list(self):
        issue_data = pd.read_csv("../files/repo_commit.csv", encoding="utf-8").drop_duplicates()
        for rows in issue_data.iterrows():
            self.repo_name_list.append(rows[1][1])
            self.issue_num_list.append(rows[1][2])
            self.user_id_list.append(rows[1][5])
            self.user_login_list.append(rows[1][6])
            self.issue_title_list.append(rows[1][3])
            if type(rows[1][4]) != str:
                rows[1][4] = ""
            rows[1][4] = rows[1][4].replace('\\', "")
            self.issue_body_list.append(rows[1][4])
            self.user_closed_id.append(rows[1][7])
            self.user_closed_login.append(rows[1][8])
            if rows[1][9] != "[]":
                rows[1][9] = rows[1][9].replace('\\', "")
                self.issue_assignees_list.append(eval(rows[1][9]))
            else:
                self.issue_assignees_list.append({})
            if rows[1][11] != "[]":
                self.issue_comment_list.append(rows[1][11].replace('# [Codecov]', ""))
            else:
                self.issue_comment_list.append("{}")
            self.issue_label_list.append(rows[1][12])
            self.repo_readme_list.append(rows[1][13])
            self.repo_language_list.append(rows[1][14])
            self.repo_topic_list.append(rows[1][15])
            self.repo_description_list.append(rows[1][16])
            if rows[1][17] != "[]":
                self.repo_contributors_list.append(rows[1][17].replace('\\n', ""))
            else:
                self.repo_contributors_list.append({})

    def get_brief_title(self):
        stop_words = self.get_stopwords()
        new_title_list = list()
        lemma = WordNetLemmatizer()
        for title in self.issue_title_list:
            lan = langid.classify(title)
            new_title = []
            if lan[0] == 'zh':
                title = jieba.lcut(title, cut_all=False)
            else:
                title = nltk.word_tokenize(title)
            for w in title:
                stop = set(stopwords.words('english'))
                if (w not in stop_words) & (w not in stop):
                    new_title.append(lemma.lemmatize(w))
            new_title_list.append(new_title)
        return new_title_list

    def get_brief_body(self):
        stop_words = self.get_stopwords()
        new_body_list = list()
        lemma = WordNetLemmatizer()
        for body in self.issue_body_list:
            body = body.replace("\\n", "")
            body = body.replace("\\r", "")
            lan = langid.classify(body)
            new_body = []
            if lan[0] == 'zh':
                body = jieba.lcut(body, cut_all=True)
            else:
                body = nltk.word_tokenize(body)
            for w in body:
                w.replace("-", " ").replace("__", "").replace("\\", " ").replace("/", " ").split(" ")
                stop = set(stopwords.words('english'))
                if type(w) is str:
                    if (w not in stop_words) & (w not in stop):
                        new_body.append(lemma.lemmatize(w))
                else:
                    for i in w:
                        if (i not in stop_words) & (i not in stop):
                            new_body.append(lemma.lemmatize(w))
            new_body_list.append(new_body)
        return new_body_list

    # def get_brief_comments(self, b):
    #     stop_words = self.get_stopwords()
    #     lan = langid.classify(b)
    #     new_b = []
    #     if lan[0] == 'zh':
    #         b = jieba.lcut(b, cut_all=True)
    #     else:
    #         b = nltk.word_tokenize(b)
    #     for w in b:
    #         w.replace("-", " ").replace("__", "").replace("\\", " ").replace("/", " ").split(" ")
    #         if type(w) is str:
    #             if w not in stop_words:
    #                 new_b.append(w)
    #         else:
    #             for i in w:
    #                 if i not in stop_words:
    #                     new_b.append(w)
    #     return new_b

    def get_brief_readme(self):
        stop_words = self.get_stopwords()
        new_readme_list = list()
        lemma = WordNetLemmatizer()
        for body in self.repo_readme_list:
            if type(body) != "str":
                new_readme_list.append(body)
            else:
                body = body.replace("\\n", "").replace("\\r", "")
                lan = langid.classify(body)
                new_body = []
                if lan[0] == 'zh':
                    body = jieba.lcut(body, cut_all=False)
                else:
                    body = nltk.word_tokenize(body)
                for w in body:
                    stop = set(stopwords.words('english'))
                    if (w not in stop_words) & (w not in stop):
                        new_body.append(lemma.lemmatize(w))
                new_readme_list.append(new_body)
        return new_readme_list

    def get_issue(self):
        issue_list = list()
        title_list = self.get_brief_title()
        body_list = self.get_brief_body()
        read_me_list = self.get_brief_readme()
        # 构造切句issue
        for i in range(len(self.user_id_list)):
            repo = Repo(self.repo_name_list[i], self.repo_language_list[i], read_me_list[i],
                              self.repo_topic_list[i],
                              self.repo_description_list[i], self.repo_contributors_list[i])
            issue = IssueEntity(self.issue_num_list[i], title_list[i], body_list[i],
                                User(self.user_id_list[i], self.user_login_list[i]),
                                User(self.user_closed_id[i], self.user_closed_login[i]),
                                self.issue_assignees_list[i], self.issue_label_list[i], repo)
            issue_list.append(issue)
        return issue_list

    # 初步获取comment
    # def get_comments(self):
    #     comment_list = list()
    #     for c in range(len(self.issue_comment_list)):
    #         repoNum = self.repo_name_list[c] + " " + str(self.issue_num_list[0])
    #         self.issue_comment_list[c] = self.issue_comment_list[c].replace('\\n', "")
    #         self.issue_comment_list[c] = self.issue_comment_list[c].replace('\\', "")
    #         # self.issue_comment_list[c].replace('\\r', "")
    #         # self.issue_comment_list[c].replace('\\t', "")
    #         if self.issue_comment_list[c] == "{}":
    #             continue
    #         if self.issue_comment_list[c].count("id") > 1:
    #             com = self.issue_comment_list[c].split("'id':")
    #             for j in com:
    #                 if j != "[{":
    #                     comment_dict = {"num": repoNum, "comment": j}
    #                     comment_list.append(comment_dict)
    #         else:
    #             comment_list.append({"num": repoNum, "comment": self.issue_comment_list[c].replace("[{'id':", "")})
    #     return comment_list

    # def final_comment_list(self):
    #     comments = self.get_comments()
    #     comment_list = list()
    #     id_list = list()
    #     login_list = list()
    #     comment_num = list()
    #     comment_body = list()
    #     relation_list = list()
    #     for co in comments:
    #         num = co["num"]
    #         temp1 = co["comment"].split(", 'login': '")
    #         if len(temp1) == 1:
    #             continue
    #         temp2 = temp1[1].split("', 'body':")
    #         if len(temp2) == 1:
    #             continue
    #         temp3 = temp2[1].split(", 'author_association': '")
    #         if len(temp3) == 1:
    #             continue
    #         else:
    #             id_list.append(temp1[0])
    #             login_list.append(temp2[0])
    #             comment_num.append(num)
    #             comment_body.append(self.get_brief_comments(temp3[0]))
    #             relation_list.append(temp3[1][0:temp3[1].index("'")])
    #     构造切句comment
    #     for index in range(len(id_list)):
    #         comment = CommentEntity(comment_body[index], UserEntity(id_list[index], login_list[index]),
    #                                 relation_list[index], comment_num[index])
    #         comment_list.append(comment)
    #     for c in comment_list:
    #         self.df_comment[self.index_comment[0]] = c.num
    #         self.df_comment[self.index_comment[1]] = c.user.userid
    #         self.df_comment[self.index_comment[2]] = c.user.login
    #         self.df_comment[self.index_comment[3]] = str(c.body)
    #         self.df_comment[self.index_comment[4]] = c.userRelate
    #         self.save_by_csv(self.df_comment, True)

    def save_by_csv(self, data_dict, is_comment=False):
        df = pd.DataFrame(data_dict, index=[0])
        path = self.result_path
        if is_comment:
            file_path = path + "comment.csv"
        else:
            file_path = path + "issue1.csv"
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False)
        else:
            df.to_csv(file_path, mode="w")

    def save(self, data_dict, is_repo=False):
        df = pd.DataFrame(data_dict)
        path = self.result_path
        if is_repo:
            file_path = path + "trainId.csv"
        else:
            file_path = path + "train.csv"
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False)
        else:
            df.to_csv(file_path, mode="w")

    @staticmethod
    def get_stopwords():
        with open('../files/baidu_stopwords.txt') as f:
            stop_words = f.read()
        return stop_words.replace('\\n', "")

    def save_issue(self, issues_list: list):
        for i in issues_list:
            self.df_issue[self.index_issue[0]] = i.repo.repoName
            self.df_issue[self.index_issue[1]] = i.num
            self.df_issue[self.index_issue[2]] = i.user.userid
            self.df_issue[self.index_issue[3]] = i.user.login
            self.df_issue[self.index_issue[4]] = str(i.closed.userid)
            self.df_issue[self.index_issue[5]] = str(i.closed.login)
            self.df_issue[self.index_issue[6]] = str(i.title)
            self.df_issue[self.index_issue[7]] = str(i.body)
            self.df_issue[self.index_issue[8]] = str(i.assignees)
            self.df_issue[self.index_issue[9]] = str(i.labels)
            self.df_issue[self.index_issue[10]] = str(i.repo.language)
            self.df_issue[self.index_issue[11]] = str(i.repo.topic)
            self.df_issue[self.index_issue[12]] = str(i.repo.description)
            self.df_issue[self.index_issue[13]] = str(i.repo.contributors)
            self.df_issue[self.index_issue[14]] = str(i.repo.readme)
            self.save_by_csv(self.df_issue)

    def train(self):
        self.df_issue = pd.read_csv("../files/issue1(总).csv", encoding="utf-8").drop_duplicates()
        doc = list()
        repo = []
        for r in self.df_issue.iterrows():
            if type(r[1][15]) != "str":
                r[1][15] = "[]"
            if type(r[1][12]) != 'str':
                r[1][12] = '[]'
            repo.append({"repo/num":
                             r[1][1] + str(r[1][2]),
                         "repo":
                             [r[1][11]] +
                             r[1][12].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ") +
                             r[1][13].replace(",", "").replace(".", "").split(" ") +
                             r[1][14].replace("'login': '", "").replace("{", "").replace("}", "").replace("[", "").
                        replace("]", "").replace(",", "").split(" ") +
                             r[1][15].replace("\'", "").replace(".", "").replace(",", " ").replace("/", " ").replace(
                                 "[", "").replace("]", "").split(" ")})
            if r[1][6] == "[]":
                continue
            doc.append({"repo/num": r[1][1] + str(r[1][2]) + str(r[1][9].replace("'login': ", "").replace("{", "").
                                                                 replace("}", "").replace("[", "").replace("]", "").
                                                                 replace(",", "").replace("'", "").split(" ")) +
                                    str(r[1][14].replace("'login': '", "").replace("{", "").replace("}", "").replace(
                                        "[", "").replace("]", "").replace(",", "").split(" ")),
                        "issue":
                            [r[1][11]] +
                            r[1][12].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ") +
                            r[1][13].replace(",", "").replace(".", "").split(" ") +
                            r[1][7].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ")
                            + r[1][8].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ") +
                            r[1][10].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ")
                        })
        train = []
        train_item_id = []
        for i in range(len(doc)):
            train_item_id.append(doc[i]["repo/num"])
            train.append(doc[i]["issue"])
        self.save(train_item_id, True)
        # 建立词典
        dictionary = gensim.corpora.Dictionary(train)
        # 建立语料库
        corpus = [dictionary.doc2bow(text) for text in train]
        # 初始化模型
        tfidf = models.TfidfModel(corpus)
        # 获取语料库的tfidf值
        corpus_tfidf = tfidf[corpus]
        # 建立索引
        index = similarities.MatrixSimilarity(corpus_tfidf, num_features=len(dictionary.keys()))
        # 保存
        tfidf.save("../model/data.tfidf")
        dictionary.save('../files/词典.dic')
        index.save('../files/train.index')
        with open('../files/词频.txt', 'a', encoding='utf-8') as f1:
            f1.write(str(corpus))

    def train_add(self):
        self.df_issue = pd.read_csv("../files/issue2.csv", encoding="utf-8").drop_duplicates()
        doc = list()
        repo = []
        for r in self.df_issue.iterrows():
            if type(r[1][15]) != "str":
                r[1][15] = "[]"
            repo.append({"repo/num":
                             r[1][1] + str(r[1][2]),
                         "repo":
                             [r[1][11]] +
                             r[1][12].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ") +
                             r[1][13].replace(",", "").replace(".", "").split(" ") +
                             r[1][14].replace("'login': '", "").replace("{", "").replace("}", "").replace("[", "").
                        replace("]", "").replace(",", "").split(" ") +
                             r[1][15].replace("\'", "").replace(".", "").replace(",", " ").replace("/", " ").replace(
                                 "[", "").replace("]", "").split(" ")})
            if r[1][6] == "[]":
                continue
            doc.append({"repo/num": r[1][1] + str(r[1][2]) + str(r[1][9].replace("'login': ", "").replace("{", "").
                                                                 replace("}", "").replace("[", "").replace("]", "").
                                                                 replace(",", "").replace("'", "").split(" ")) +
                                    str(r[1][14].replace("'login': '", "").replace("{", "").replace("}", "").replace(
                                        "[", "").replace("]", "").replace(",", "").split(" ")),
                        "issue":
                            r[1][12].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ") +
                            r[1][13].replace(",", "").replace(".", "").split(" ") +
                            r[1][7].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ")
                            + r[1][8].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ") +
                            r[1][10].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(" ")
                        })
        train = []
        train_item_id = []
        for i in range(len(doc)):
            train_item_id.append(doc[i]["repo/num"])
            train.append(doc[i]["issue"])
        # self.save(train_item_id, True)
        dictionary = gensim.corpora.Dictionary.load('../files/词典.dic')
        dictionary.add_documents(train)
        with open("../files/词频.txt", "r", encoding="utf-8") as f:
            corpus = f.read()
            list1 = corpus.split('], [')
            corpus_list = []
            for each in list1:
                each = each.replace('[[', '').replace(']]', '')
                # print(each)
                each_list = each.split('), (')
                each_list1 = []
                for item in each_list:
                    # print(item)
                    item = item.replace('(', '').replace(')', '').replace(' ', '')
                    item = eval(item)  # (1, 2)
                    # print(item)
                    each_list1.append(item)  # [(), (), ……, ()]
                # print(each_list1)
                corpus_list.append(each_list1)
        # corpus_list += [dictionary.doc2bow(text) for text in train]
        tfidf = models.TfidfModel(corpus_list, id2word=dictionary)
        # 获取语料库的tfidf值
        corpus_tfidf = tfidf[corpus_list]
        # 建立索引
        index = similarities.MatrixSimilarity(corpus_tfidf, num_features=len(dictionary.keys()))
        # 保存
        # tfidf.save("../model/data.tfidf")
        # dictionary.save('../files/词典.dic')
        # index.save('../files/train.index')
        # with open('../files/词频.txt', 'a', encoding='utf-8') as f1:
        #     f1.write(str(corpus))


if __name__ == '__main__':
    wm = WordsManager()
    # wm.issue_body_list = ["Fix networks initialization"
    #                       "Currently, we trigger all the pod and endpoint read events. This always fails, and spams the "
    #                       "logs, since Services have not been synced.Instead, at startup just prepare the "
    #                       "networks/gateways and rely on the standard SyncAll call to trigger the explicit pod "
    #                       "sync.Without this, we get many `Handle EDS endpoint: skip collecting workload entry "
    #                       "endpoints, service rancher.io-local-path/local-path-storage has not been. populated`"]
    # print(wm.get_brief_body())
    # 分词
    # wm.set_list()
    # wm.final_comment_list()
    # wm.save_issue(wm.get_issue())

    # readIssue
    wm.train()
