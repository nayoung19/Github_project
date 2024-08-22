import random
import os
import json
import pandas as pd


class Util(object):
    def __init__(self):
        self.config_path = os.path.dirname(__file__) + os.sep
        f = open(self.config_path + "config.json", mode="r", encoding="utf-8")
        self.json_result = json.load(f)
        self.result_path = '../files'

    def get_user_agent(self):
        if "user_Agent" in self.json_result:
            user_agent_list = self.json_result["user_Agent"]
            user_agent = random.choice(user_agent_list)
            return user_agent
        else:
            print("请在config.json中配置user_Agent。")

    # 获取线程数量
    def get_thread_num(self):
        if "thread_num" in self.json_result:
            thread_num = self.json_result["thread_num"]
            return thread_num
        else:
            print("请在config.json中配置thread_num。")

    # 获取数据批量保存一次的数量
    def get_batch_num(self, is_repo=False):
        if is_repo:
            if "repo_batch_size" in self.json_result:
                batch_size = self.json_result["repo_batch_size"]
                return batch_size
            else:
                print("请在config.json中配置repo_batch_size。")
        else:
            if "user_batch_size" in self.json_result:
                batch_size = self.json_result["user_batch_size"]
                return batch_size
            else:
                print("请在config.json中配置user_batch_size。")

    def get_users(self):
        userdata = pd.read_csv(self.result_path + "repo_commit.csv", encoding="utf-8")
        user_name_list = list()
        for rows in userdata.iterrows():
            if (rows[1][6] != "[]") & (rows[1][8] != "[]"):
                user_name_list.append(rows[1][6])
                user_name_list.append(rows[1][8])
        return set(user_name_list)

    def get_users_by_comments(self):
        userdata = pd.read_csv(self.result_path + "repo_commit.csv", encoding="utf-8")
        user_name_list = list()
        for rows in userdata.iterrows():
            if rows[1][11] != "[]":
                for i in eval(rows[1][11]):
                    user_name_list.append(i["login"])
        return set(user_name_list)

    def get_users_by_contributor(self):
        userdata = pd.read_csv(self.result_path + "repo_commit.csv", encoding="utf-8")
        user_name_list = list()
        for rows in userdata.iterrows():
            if rows[1][17] != "[]":
                for i in eval(rows[1][17]):
                    user_name_list.append(i["login"])
        return set(user_name_list)

    @staticmethod
    def get_users_login():
        user_name_list = list()
        user_name_list.append(util.get_users_by_contributor())
        user_name_list.append(util.get_users_by_comments())
        user_name_list.append(util.get_users())
        return user_name_list

    # 获取github api token
    def get_api_token(self):
        if "api_token" in self.json_result:
            api_token = self.json_result["api_token"]
            if api_token != "":
                return api_token
            print("请在config.json中配置api_token。")
        else:
            print("请在config.json中配置api_token。")

    def save(self, data_dict, is_repo=False):
        if "save_method" in self.json_result:
            save_method = self.json_result["save_method"]
            if save_method == "csv":
                self.__save_by_csv(data_dict, is_repo)
            elif save_method == "mongodb":
                self.__save_by_mongodb(data_dict)
            else:
                print("请在config.json中配置user_Agent。")

    def __save_by_mongodb(self, data_dict):
        pass

    def __save_by_csv(self, data_dict, is_repo=False):
        df = pd.DataFrame(data_dict)
        if is_repo:
            file_path = "../files/repo_commit.csv"
        else:
            file_path = "../files/info.csv"
        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False)
        else:
            df.to_csv(file_path, mode="w")

    def process_bar(self, percent, start_str='', end_str='', total_length=0):
        bar = '\r' + start_str + ''.ljust(int(percent * total_length), "=") + '> {:0>5.2f}%|'.format(
            percent * 100) + end_str
        print(bar, end='', flush=True)


if __name__ == '__main__':
    util = Util()
    # print(util.get_users())
    # util.save_by_mysql()
    # print(util.get_users_by_comments())
    # print(util.get_users_by_contributor())
    # util.get_users_by_contributor()
