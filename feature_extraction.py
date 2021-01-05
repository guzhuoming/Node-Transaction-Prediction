"""
features extraction:
transaction num, transaction sum, transaction mean, transaction variance
"""
import pandas as pd
import csv
import numpy as np
import os

def feature_extraction(ts = 80, minTime=1590940800, n=1):
    """

    :return:
    """
    file = open('./address.csv')
    df = pd.read_csv(file)
    address = df['address']

    # create feature files
    for i in range(len(address)):
        if not os.path.exists('./data/feature_4_{}'.format(n)):
            os.makedirs('./data/feature_4_{}'.format(n))
        file2 = open('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]), 'w',
                    newline='')
        csvwriter = csv.writer(file2)
        csvwriter.writerow(['tran_num', 'tran_sum', 'tran_mean', 'tran_var'])

        for j in range(ts):
            csvwriter.writerow([0. for i in range(4)])
        file2.close()

    for i in range(len(address)):
        print('i={}'.format(i))
        node = address[i]
        data = open('./data/source_data/{}.csv'.format(node))
        df_data = pd.read_csv(data)

        # save transaction values
        tran = [[] for i in range(ts)]

        for j in range(len(df_data)):
            ft = open('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]))
            df_ft = pd.read_csv(ft)
            ft.close()

            t = (df_data['TimeStamp'][j] - minTime) // (86400 * n)
            if n==1:
                t-=21
            elif n==3:
                t-=7
            if t>=0 and t<ts :
                df_ft['tran_num'][t] = df_ft['tran_num'][t] + 1
                df_ft['tran_sum'][t] = df_ft['tran_sum'][t] + df_data['Value'][j]

                tran[t].append(df_data['Value'][j])

            df_ft.to_csv('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]),
                         index=False)
        for t in range(ts):
            if len(tran[t])>0:
                df_ft['tran_mean'][t] = np.mean(tran[t])
                df_ft['tran_var'][t] = np.var(tran[t])

            else:
                df_ft['tran_mean'][t] = 0
                df_ft['tran_var'][t] = 0

        df_ft.to_csv('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]),
                     index=False)

if __name__=='__main__':
    feature_extraction(ts = 80, minTime=1590940800, n=1)