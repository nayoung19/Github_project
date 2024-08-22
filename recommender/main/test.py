import pandas as pd

from recommender.main.main2 import Main2
from tqdm import tqdm


class Test(object):
    @staticmethod
    def topK(test_issue):
        n = 0
        for issue in tqdm(test_issue):
            m = Main2()
            m.get_similar_issue(issue['content'])
            for item in m.idlist:
                if item == issue['issue_name']:
                    n += 1
                    print(n)
                    break
            pass
        return n

    @staticmethod
    def get_test():
        df = pd.read_csv("../files/repo_commit_test.csv", encoding='utf-8')
        issue_list = []
        for row in df.iterrows():
            if row[1][7] == "[]":
                continue
            if type(row[1][4]) != 'str':
                row[1][4] = ''
            issue_info = {'issue_name': row[1][1] + str(row[1][2]),
                          'content': row[1][3] + row[1][4]}
            issue_list.append(issue_info)
        return issue_list


if __name__ == '__main__':
    t = Test()
    testList = t.get_test()
    total_num = len(testList)
    success_num = t.topK(testList)
    print(total_num)
    print(success_num)
    print(success_num / total_num)
