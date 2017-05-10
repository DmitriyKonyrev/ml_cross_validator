import matplotlib.pyplot as plt
import os
import subprocess as sp
import numpy as np
import itertools

class CategoryGraphicsHolder:
    def __init__(self, category_path, positive_count, blur_factor):
        self.category_name  = os.path.basename(category_path)
        self.category_path  = category_path
        self.positive_count = positive_count
        self.blur_factor    = blur_factor

        self.learning_curves = {
              'logloss' : []
            , 'rmse'    : [],
        }

        self.metrics_detailed = {
            # learn set params
        	  'learn_f1'              : []
        	, 'learn_precision'       : []
        	, 'learn_complete'        : []
        	, 'learn_accuracy'        : []
        	, 'learn_rmse'            : []
            # test set params
        	, 'test_f1'               : []
        	, 'test_precision'        : []
        	, 'test_complete'         : []
        	, 'test_accuracy'         : []
        	, 'test_rmse'             : []
            # model params
        	, 'model_complexity'      : []
            , 'time'                  : [],
        }

        self.metric_data = {
            # learning curves params
        	  'average_learn_logloss' : 0.0
        	, 'average_learn_rmse'    : 0.0
            # learn set params
        	, 'learn_f1'              : 0.0
        	, 'learn_precision'       : 0.0
        	, 'learn_complete'        : 0.0
        	, 'learn_accuracy'        : 0.0
        	, 'learn_rmse'            : 0.0
            # test set params
        	, 'test_f1'               : 0.0
        	, 'test_precision'        : 0.0
        	, 'test_complete'         : 0.0
        	, 'test_accuracy'         : 0.0
        	, 'test_rmse'             : 0.0
            # model params
        	, 'model_complexity'      : 0.0
            , 'time'                  : 0.0,
        }
        self.parse_data()


    def __parse_metrics(self, metric_name):
        metric_file = os.path.join(self.category_path, metric_name + "_path")
        for line in open(metric_file, 'r'):
            value = 0.0 if 'nan' in line else float(line)
            self.metrics_detailed[metric_name].append(value)
            self.metric_data[metric_name] = value


    def __parse_leaning_curves(self, curve_name):
        curve_file = os.path.join(self.category_path, "learning_" + curve_name)
        curve_average = 0.0
        curve_points  = 0
        for line in open(curve_file, 'r'):
            value = 0.0 if 'nan' in line else float(line)
            self.learning_curves[curve_name].append(value)
            curve_average += value
            curve_points  += 1
        curve_average /= curve_points
        self.metric_data['average_learn_{0}'.format(curve_name)]


    def __build_base_metrics(self, metric_name, factor_name, outpath):
        filename = os.path.join(outpath, 'graphics_{0}_{1}.png'.format(factor_name, metric_name))
        plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.xlabel(factor_name)
        plt.ylabel(metric_name)
        plt.title('Category {0} {1}-{2}: average learn {3} - test {4}'
                  .format( os.path.basename(self.category_path)
                         , factor_name
                         , metric_name
                         , self.metric_data['learn_' + metric_name]
                         , self.metric_data['test_'  + metric_name])
                  )
        lists = sorted(zip(*[self.metrics_detailed[factor_name], self.metrics_detailed['learn_' + metric_name]]))
        new_x, new_y = list(zip(*lists))
        plt.plot(new_x, new_y, 'b+-', label=('learn_' + metric_name))
        lists = sorted(zip(*[self.metrics_detailed[factor_name], self.metrics_detailed['test_' + metric_name]]))
        new_x, new_y = list(zip(*lists))
        plt.plot(new_x, new_y, 'ro-', label=('test_' + metric_name))
        plt.grid(True)
        plt.savefig(filename)


    def __build_learning_curve(self, curve_name, outpath):
        filename = os.path.join(outpath, 'graphics_learning_curve_{0}.png'.format(curve_name))
        plt.figure(num=None, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.xlabel('iterations x 10^2')
        plt.ylabel(curve_name)
        plt.title('Category {0} learning curve {1}: average {2}'
                  .format( os.path.basename(self.category_path)
                         , curve_name
                         , self.metric_data['average_learn_' + curve_name])
                  )
        plt.plot(list(range(len(self.learning_curves[curve_name]))), self.learning_curves[curve_name], 'b+-', label=('learning ' + curve_name))
        plt.grid(True)
        plt.savefig(filename)


    def parse_data(self):
        for set_type in ['learn', 'test']:
            for metric_type in ['f1', 'precision', 'complete', 'accuracy', 'rmse']:
                self.__parse_metrics(set_type + "_" + metric_type)
        self.__parse_metrics('model_complexity')
        self.__parse_metrics('time')
        self.__parse_leaning_curves('logloss')
        self.__parse_leaning_curves('rmse')


    def build_graphics(self, outpath):
        out_dir = os.path.join(outpath, os.path.basename(self.category_path))
        sp.check_call(['mkdir', out_dir])
        for factor_type in ['time', 'model_complexity']:
            for metric_type in ['f1', 'precision', 'complete', 'accuracy', 'rmse']:
                self.__build_base_metrics(metric_type, factor_type, out_dir)
        self.__build_learning_curve('logloss', out_dir)
        self.__build_learning_curve('rmse', out_dir)


class ClassifierGraphicsHolder:
    def __init__(self, classifier_path):
        self.classifier_path = classifier_path
        self.classfier_name = os.path.basename(classifier_path)
        self.classifier_data = list()
        self.__parse_data()


    def __parse_data(self):
        for line in open(os.path.join(self.classifier_path, 'categories_data'), 'r'):
            category, volume, blur = line.split('\t')
            volume = float(volume)
            blur   = float(blur)
            holder = CategoryGraphicsHolder(os.path.join(self.classifier_path, category), volume, blur)
            self.classifier_data.append(holder)


    def __build_averages_bars(self, outpath, metric_name, factor_type):
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%f' % height,
                        ha='center', va='bottom')
        filename = os.path.join(outpath, 'graphic_averge_{0}_{1}_{2}.png'.format(self.classfier_name, metric_name, factor_type))
        plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.xlabel('categories {0}'.format(factor_type))
        plt.ylabel(metric_name)
        plt.title('Classifier {0} average {1} by {2}'
                  .format( self.classfier_name
                         , metric_name
                         , factor_type)
                  )
        values = [(  category_data.category_name
                   , category_data.positive_count if factor_type == 'volume' else category_data.blur_factor
                   , category_data.metric_data['learn_' + metric_name]
                   , category_data.metric_data['test_' + metric_name])
                  for category_data in self.classifier_data]
        values = sorted(values, reverse=True, key=lambda v: v[1])
        names = ['{0}:{1}'.format(v[0], v[1]) for v in values]
        learn = [v[2] for v in values]
        test  = [v[3] for v in values]
        bar_width = 0.35
        opacity = 0.4
        index = np.arange(len(names))
        rects1 = plt.bar( index, learn
                        , bar_width
                        , alpha=opacity
                        , color='b'
                        , label='learn')
        rects2 = plt.bar( index + bar_width, test
                        , bar_width
                        , alpha=opacity
                        , color='r'
                        , label='test')
        plt.xticks(index + bar_width / 2, names)
        plt.legend()
        plt.tight_layout()
        autolabel(rects1)
        autolabel(rects2)
        plt.savefig(filename)


    def build_graphics(self, outpath):
        out_dir = os.path.join(outpath, self.classfier_name)
        sp.check_call(['mkdir', out_dir])
        for category_data in self.classifier_data:
            category_data.build_graphics(out_dir)
        for metric_name in ['f1', 'precision', 'complete', 'accuracy', 'rmse']:
            for factor_type in ['volume', 'blur_factor']:
                self.__build_averages_bars(out_dir, metric_name, factor_type)
