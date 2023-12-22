import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import statsmodels.api as sm
import scipy.stats



csv_dir = "/home/jjxia/Documents/projects/All_Elbow/out/csv_sobels"
plot_save_dir = "/home/jjxia/Documents/projects/All_Elbow/out/csv_plot"


corre_r = []
x_label = []

# https://realpython.com/numpy-scipy-pandas-correlation-python/
def cal_correlation(x, y):
    r, p = scipy.stats.pearsonr(x, y)
    # r = np.corrcoef(x, y)
    # print("r ", r)
    # print("p ", p)
    return r
    

def plot_xy(x, y, plot_save_path):
    x = np.array(x)
    y = np.array(y)
    plt.clf()
    plt.scatter(x, y, color= "green", s=5)
    plt.xlabel('x - sobel score')
    # frequency label
    plt.ylabel('y - gradient norm')
    # plot title
    
    # showing legend
    # function to show the plot
    # b, m = polyfit(x, y, 1)
    # plt.plot(x, b + m * x, '-')
    results = sm.OLS(y,sm.add_constant(x)).fit()
    print(results.summary())
    r = cal_correlation(x, y)
    plt.title('correlation coefficient ' + str(r))

    m, b = np.polyfit(x, y, deg=1)
    plt.axline(xy1=(0, b), slope=m, color='r', label=f'$y = {m:.2f}x {b:+.2f}$')
    # X_plot = np.linspace(0,1,500)
    # plt.plot(X_plot, X_plot * results.params[1] + results.params[0])
    plt.savefig(plot_save_path)
    
# curent_file_name = ''

def plot_corre(csv_path, plot_save_path, file_name):
    data = None
    data = np.genfromtxt(csv_path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    plt.clf()
    plt.scatter(x.tolist(), y.tolist(), color= "green", s=5)
    plt.xlabel('x - sobel score')
    # frequency label
    plt.ylabel('y - gradient norm')
    # plot title
    # plt.title('Correlation between sobel score and gradient norm')
    # showing legend
    # function to show the plot
    # b, m = polyfit(x, y, 1)
    # plt.plot(x, b + m * x, '-')
    
    # compute slope m and intercept b
    m, b = np.polyfit(x, y, deg=1)

    r = cal_correlation(x, y)
    corre_r.append(r)
    plt.title(file_name + ' correlation coefficient ' + str(round(r, 3)))

    # plot fitted y = m*x + b
    plt.axline(xy1=(0, b), slope=m, color='r', label=f'$y = {m:.2f}x {b:+.2f}$')
    plt.savefig(plot_save_path)


def draw_points_all_in_one(csv_dir):
    csv_files = os.listdir(csv_dir)
    x_list = []
    y_list = []
    for file in csv_files:
        csv_file_path = os.path.join(csv_dir, file)
        data = np.genfromtxt(csv_file_path, delimiter=',')
        x = data[:, 0].tolist()
        y = data[:, 1].tolist()
        x_list.extend(x)
        y_list.extend(y)
    plot_save_path = os.path.join(plot_save_dir,  "all.png")
    plot_xy(x_list, y_list, plot_save_path)


def plot_batches():
    csv_files = os.listdir(csv_dir)
    if not os.path.exists(plot_save_dir):
        os.mkdir(plot_save_dir)
    for file in csv_files:
        csv_file_path = os.path.join(csv_dir, file)
        file_name = file.split('.')[0]
        x_label.append(file_name)
        plot_save_path = os.path.join(plot_save_dir,  file_name + ".png")
        
        plot_corre(csv_file_path, plot_save_path, file_name)
    


def plot_single_bar(valid_labels, iou_rates, plot_name):
    x = np.arange(len(valid_labels))  # the label locations
    width = 0.5  # the width of the bars
    multiplier = 0
    plt.clf()
    fig, ax = plt.subplots(layout='constrained')
    # for attribute, measurement in penguin_means.items():

    rects = ax.bar(x + width, iou_rates, width)
    bar_labels = [str(round(val, 3)) for val in iou_rates] 
    ax.bar_label(rects, labels=bar_labels, padding=0)
    multiplier += 1
    fig.set_figheight(8)
    fig.set_figwidth(18)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('correlation rate')
    ax.set_title('Sobel Score and Gradient Correlation Coefficient')
    ax.set_xticks(x + width, valid_labels)
    plt.xticks(rotation=30, ha='right')
    # ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 1)
    # plt.show()
    plt.savefig(plot_name)

if __name__ == "__main__":
    plot_batches()
    draw_points_all_in_one(csv_dir)
    print(x_label)
    print(corre_r)
    plot_single_bar(x_label, corre_r, "Correlation.png")