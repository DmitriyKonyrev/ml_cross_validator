import argparse
import csv
import os
import math
import subprocess as sp
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.externals import six
import six


from graphics_holder import *

def parse_args():
    DEFAULT_INPUTDIR        = os.getcwd()
    DEFAULT_OUTDIR          = os.path.join(os.getcwd(), 'graphics')

    parser = argparse.ArgumentParser(description='build graphics')
    parser.add_argument('-i'
                       , '--input-dir'
                       , type=str
                       , default=DEFAULT_INPUTDIR
                       , help="directory with input data (default: {0})".format(DEFAULT_INPUTDIR)
                       )
    parser.add_argument('-o'
                       , '--outdir'
                       , type=str
                       , default=DEFAULT_OUTDIR
                       , help="output directory (default: {0})".format(DEFAULT_OUTDIR)
                       )
    settings = parser.parse_args()
    if not os.path.isdir(settings.outdir):
        sp.check_call(['mkdir', settings.outdir])
    print(settings)
    return settings


def parse_data(settings):
    graphics_holders = []
    for classifier in os.listdir(settings.input_dir):
        print('read data of {0}'.format(classifier))
        graphics_holders.append(ClassifierGraphicsHolder(os.path.join(settings.input_dir, classifier)))
    return graphics_holders


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                                          header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                                          bbox=[0, 0, 1, 1], header_columns=0,
                                          ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        print(size)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax


def build_graphics(settings, graphics_holders):
    header = []
    for holder in graphics_holders:
        print('process classifier {0} data'.format(holder.classfier_name))
        header.append(holder.classfier_name)
        holder.build_graphics(settings.outdir)
    print('build result table')
    metrics = [ 'learn_f1',        'test_f1'
              , 'learn_precision', 'test_precision'
              , 'learn_complete',  'test_complete'
              , 'learn_accuracy',  'test_accuracy'
              , 'learn_rmse',      'test_rmse'
              , 'average_learn_logloss'
              , 'average_learn_rmse'
              , 'time'
              , 'model_complexity']
    classifier_index = ['average' for i in range(len(metrics))]
    classifier_metrics_index = []
    classifier_metrics_index += metrics
    for holder in graphics_holders[0].classifier_data:
       classifier_index += [holder.category_name for i in range(len(metrics))]
       classifier_metrics_index += metrics
    index = pd.MultiIndex.from_tuples(list(zip(classifier_index, classifier_metrics_index)), names=['category', 'metric'])
    table = pd.DataFrame(0.0, index=index, columns=header)
    for holder in graphics_holders:
        for categories_holder in holder.classifier_data:
            for metric in metrics:
                table.set_value((categories_holder.category_name, metric), holder.classfier_name, categories_holder.metric_data[metric])
                table[holder.classfier_name][('average', metric)] += categories_holder.metric_data[metric]
        for metric in metrics:
            table[holder.classfier_name][('average', metric)] /= float(len(holder.classifier_data))
    html_out = os.path.join(settings.outdir, 'classifiers.html')
    png_out  = os.path.join(settings.outdir, 'classifiers.png')
    with open(html_out, 'w') as html_outfile:
        styler = table.style.highlight_max(color='green',axis=1)
        if len(graphics_holders) > 1:
            styler = styler.highlight_max(color='red',axis=1)
        html_outfile.write(styler.render())
    sp.check_call(['cutycapt', '--url=file:{0}'.format(html_out),
        '--out={0}'.format(png_out)])


if __name__ == '__main__':
    settings = parse_args()
    graphics_holders = parse_data(settings)
    build_graphics(settings, graphics_holders)
