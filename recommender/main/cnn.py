from gensim.models import word2vec
from keras import Input, Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, concatenate, Embedding
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from recommender.Requester.ReposRequester import ReposRequester
from recommender.Requester.UserRequester import UserRequester


class Cnn:

    @staticmethod
    def train():
        user = []
        level = []
        # score1 = pd.read_csv('../files/user_ability.csv', encoding='utf-8')
        # score2 = pd.read_csv('../files/read_ability.csv', encoding='utf-8')
        # levels = pd.read_csv('../files/user_score.csv', encoding='utf-8')
        # for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        #     user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][6], r2[1][2], r2[1][3], r2[1][4]])
        #     level.append(r3[1][3])
        # for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        #     user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][7], r2[1][2], r2[1][3], r2[1][4]])
        #     level.append(r3[1][5])
        # for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        #     user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][8], r2[1][2], r2[1][3], r2[1][4]])
        #     level.append(r3[1][7])
        # for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        #     user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][9], r2[1][2], r2[1][3], r2[1][4]])
        #     level.append(r3[1][9])
        # for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        #     user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][10], r2[1][2], r2[1][3], r2[1][4]])
        #     level.append(r3[1][11])
        # for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        #     user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][11], r2[1][2], r2[1][3], r2[1][4]])
        #     level.append(r3[1][13])
        for row in pd.read_csv("../files/smote.csv").iterrows():
            user.append([row[1][0], row[1][1], row[1][2], row[1][3], row[1][4], row[1][5], row[1][6], row[1][7]])
            level.append([row[1][8]])
        train_data = np.array(user)
        train_data = train_data.reshape(train_data.shape[0], 8, )
        x_lable = np.array(level)
        le = LabelEncoder()
        trainLable = le.fit_transform(x_lable)
        trainLable = to_categorical(trainLable)
        xs = Input(shape=(8, 1), dtype='float64')
        start = timer()
        cnn1 = Conv1D(16, 2, strides=1, activation='relu', input_shape=(8, 1))(xs)
        cnn1 = MaxPooling1D(pool_size=2, strides=1)(cnn1)
        f = Flatten()(cnn1)
        drop = Dropout(0.2)(f)
        d = Dense(64, activation='relu')(drop)
        main_output = Dense(5, activation='softmax', name='color_output')(d)
        model = Model(inputs=xs, outputs=main_output)
        print('out:')
        print(main_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        num_epochs = 180
        # history=model.fit(train_data, trainLable, epochs=num_epochs, batch_size=16, validation_split=0.2)
        history = model.fit(x=train_data, y=trainLable, validation_split=0.2, epochs=num_epochs, batch_size=64,
                            verbose=1)
        print(timer() - start)
        # 保存模型
        # model.save("../model/CNN")
        plt.style.use("ggplot")  # matplotlib的美化样式
        plt.figure()
        N = num_epochs
        plt.plot(np.arange(0, N), history.history['loss'],
                 label="train_loss")  # model的history有四个属性，loss,val_loss,acc,val_acc
        plt.plot(np.arange(0, N), history.history['val_loss'], label="val_loss")
        plt.plot(np.arange(0, N), history.history['accuracy'], label="train_acc")
        plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
        plt.title("loss and accuracy")
        plt.xlabel("epoch")
        plt.ylabel("loss/acc")
        plt.legend(loc="best")
        # plt.savefig("../files/result.png")
        plt.show()

    @staticmethod
    def evaluate(username, language):
        c = Cnn()
        u = UserRequester()
        r = ReposRequester()
        user_info = u.get_single_user_info(username)
        capacity = {"repo_num": 0, language: 0, "code": 0, "stargazers_count": 0, "commit_num": 0}
        read = r.get_fork_commit(username)
        for reponame in user_info["repos"]:
            detail = r.get_repo_details(username, reponame)
            capacity["repo_num"] += 1
            capacity["code"] += detail["code"]
            capacity["stargazers_count"] += detail["stargazers_count"]
            capacity["commit_num"] += detail["commit_num"]
            keys = list(detail["language"].keys())
            if language in keys:
                value = detail["language"][language]
                capacity[language] += value
            else:
                capacity[language] += 0
        test_data = [capacity["repo_num"], capacity["code"], capacity["stargazers_count"], capacity["commit_num"],
                     capacity[language], read["fork_num"], read["star"], read["commit"]]
        print(test_data)
        testSeq = np.array([test_data])
        # 预测与评估
        mainModel = load_model('../model/CNN')
        result = mainModel.predict(testSeq)  # 预测样本属于每个类别的概率
        # 概率读取
        lable = [1, 2, 3, 4, 5]
        count = 0
        for i in result[0]:
            percent = '%.2f%%' % (i * 100)
            print(lable[count], '', percent)
            count += 1


if __name__ == '__main__':
    c = Cnn()
    c.train()
    # c.evaluate("WiNloSt", 'Objective-C')


