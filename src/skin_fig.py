#!/usr/bin/env python3

# standard library
import io
from pathlib import Path

# common numerical and scientific libraries
import pandas as pd
import imageio
import skimage.color

# plotting
import matplotlib as mpl
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import inset_locator

# other common libraries
import dateutil
import joblib
joblib_memory = joblib.Memory('cache/skin', verbose=2)
import cv2  # opencv

# local
import scriptlib as scr
from metrics import eval_metrics, PrintMetrics
#from probability import ComputeProbability
from util import read_csv

# global parameters
colors = rcParams['axes.prop_cycle'].by_key()['color']
colors = colors * 10

@scr.skip
class DisplayTrain(scr.SubCommand):
    name = 'display-train'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--ylim',
                default='None,None',
                help='ylim values: ymin,ymax. Default: %(default)s')
        parser.add_argument('--time',
                action='store_true',
                help='x axis: train time in s. Default: epoch')
        parser.add_argument('logfile',
                nargs='+',
                help='logfiles')
            
    def run(self):
        p = self.p
        for i, logfile in enumerate(p.logfile):
            df = pd.read_csv(logfile, delim_whitespace=True)
            df['time'] = df['date_time'].map(lambda x_:
                    dateutil.parser.parse(x_.replace('_', 'T')).timestamp())
            #df = df.iloc[3:]
            if p.time:
                time_start = 2 * df.iloc[0]['time'] - df.iloc[1]['time']
                x = (df['time']-time_start)/3600
            else:
                x = np.arange(len(df)) + 1
            plot(x, df.tr_loss, 'o--', 
                    color=colors[i])
            plot(x, df.va_loss, 'o-',
                    color=colors[i], label=Path(logfile).stem)
        try:
            ymin = float(p.ylim.split(',')[0])
        except:
            ymin = None
        try:
            ymax = float(p.ylim.split(',')[1])
        except:
            ymax = None
        ylim(ymin, ymax)
        ylabel('loss')
        xlabel('train time [hr]')
        legend()
        show()

@scr.skip
class DisplayCMTrain(scr.SubCommand):
    name = 'display-cm-train'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('logfile',
                help='logfile')

    def run(self):
        p = self.p
        collection = []
        current = None
        for line in open(p.logfile):
            if line.startswith('ep'):
                if current:
                    collection.append(current)
                current = line
            else:
                current += line
        if current:
            collection.append(current)
        fig, (ax0, ax1) = subplots(2, 1, figsize=(6,10))
        for i, text in enumerate(collection):
            df = pd.read_csv(io.StringIO(text), delim_whitespace=True)
            sca(ax0)
            plot(df['ep'], df['va_loss'],
                    '-', color=colors[i])
            plot(df['ep'].iloc[1:], df['tr_loss'].iloc[1:],
                    '--', color=colors[i])
            xlabel('epoch')
            ylabel('loss')
            title(p.logfile)
            #
            sca(ax1)
            plot(df['ep'], df['va_ba'],
                    '-', color=colors[i])
            xlabel('epoch')
            ylabel('ba')
            ylim(.926, .942)

        show()
        print(len(collection))

@scr.skip
class DisplayLRDependence(scr.SubCommand):
    name = 'd-lrdep'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('outfile',
                nargs='+',
                help='outfiles')

    def run(self):
        p = self.p
        fig, (ax0, ax1) = subplots(2, 1, figsize=(6,8))
        xmin = 1e100
        xmax = -1
        labels = ['']*10
        labels = ['CM random init', 'CM avg init']
        for i, outfile in enumerate(p.outfile):
            df = read_csv(outfile, delim_whitespace=True)
            xmin = min(xmin, df['model'].min())
            xmax = max(xmax, df['model'].max())
            sca(ax0)
            semilogx(df['model'], df['AUROC'], 'o-', label=labels[i])
            ylabel('AUROC')
            sca(ax1)
            semilogx(df['model'], df['th_BA'], 'o-', label=labels[i])
            ylabel('th_BA')
        sca(ax0)
        semilogx([xmin*.5, xmax], [.9841]*2, 'r-', label='softmax average')
        legend()
        sca(ax1)
        semilogx([xmin*.5, xmax], [.8912]*2, 'r-', label='softmax average')
        legend()
        xlabel('learning rate')
        show()



class Eval(scr.SubCommand):
    name = 'eval'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--proba-parameters',
                help='model_proba parameters')
        parser.add_argument('--proba-model',
                help='proba model.h5')
        parser.add_argument('--output',
                help='optional: path to output (default: stdout)')
        parser.add_argument('--labels',
                help='optional: comma sep model labels. Default: filename stem')
        parser.add_argument('--latex',
                action='store_true',
                help='latex table output')
        parser.add_argument('result_csv',
                nargs='+',
                help='path to result csv file (name, true, pred)')

    def run(self):
        threshold = 0.5
        #
        p = self.p

        if p.output:
            outf = open(p.output, 'a')
            stdout_orig = sys.stdout
            sys.stdout = outf
            print('-'*79)

        labels = None
        if p.labels:
            labels = p.labels.split(',')
            assert len(labels) == len(p.result_csv)
        else:
            # latest path segment without suffix which is different
            tmp = []
            for pp in p.result_csv:
                pp = Path(pp)
                tmp.append([pp.stem] + [p_.name for p_ in pp.parents])
            tmp = np.array(tmp)
            for i in range(tmp.shape[1]):
                if tmp.shape[0] == 1:
                    break
                if not np.all(tmp[0,i] == tmp[:,i]):
                    break
            else:
                labels = tmp[:,0]
            labels = tmp[:,i]



        # metrics_softmax
        print('# Softmax')
        if p.latex:
            pm_args = {'sep':'&', 'end':'\\\\'}
        else:
            pm_args = {}
        pm = PrintMetrics(**pm_args)
        for i, result_csv in enumerate(p.result_csv):
            label = labels[i][-18:]
            yt = read_csv(result_csv)['true']
            yp = read_csv(result_csv)['pred']
            dat = eval_metrics(yt, yp, threshold)
            pm.append(f'{label}', dat)
        print()

        # metrics_proba
        if p.proba_parameters is None and p.proba_model is None:
            return
        print('# Calibrated probability')
        pm = PrintMetrics()
        if p.proba_model is not None:
            model_proba = ComputeProbability(p.proba_model)
        else:
            model_proba = ComputeProbability()
            model_proba.set(
                    *[float(x_) for x_ in p.proba_parameters.split(',')])
        for result_csv in p.result_csv:
            label = Path(result_csv).stem
            yt = read_csv(result_csv)['true']
            yp = read_csv(result_csv)['pred']
            proba = model_proba.predict(yp)
            dat = eval_metrics(yt, proba, model_proba.opt_ba_thres)
            pm.append(f'{label}', dat)

        if p.output:
            outf.close()
            sys.stdout = stdout_orig

class EvalDsetdep(scr.SubCommand):
    name = 'eval-dsetdep'
    threshold = 0.5

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--model-n',
                type=int, required=True,
                help='model number, eg. 21 or 41')
        parser.add_argument('--seeds',
                default='42,43,44,45,46',
                help='rng seeds, default: %(default)s')
        parser.add_argument('--latex',
                action='store_true',
                help='latex output')
        parser.add_argument('version',
                help='version, eg. cv_wt_aug')

    def run(self):
        p = self.p
        res = self.calc(p.model_n, p.version, p.seeds)
        if p.latex:
            selects = ['ba_train', 'ba_other', 'bacp_train', 'bacp_other',]
            for select in selects:
                print(f'& {100*res[select]:.1f}', end='')
                print(f'\\err{{{100*res[select+"_std"]:.2f}}} ', end='')
            print('\\\\')
        else:
            selects = ['ba_train', 'ba_other', 'bacp_train', 'bacp_other',
                    'roc_train', 'roc_other']
            print(f'{"model":18s}BA_tra BA_oth  BAcp_t BAcp_o  ROC_t  ROC_o')
            print(f'{p.version:18s}', end='')
            for select in selects:
                print(f'{res[select]:.3f}  ', end='')
            print()
            print(f'{"error":18s}', end='')
            for select in selects:
                print(f'{res[select+"_std"]:.4f} ', end='')
            print()
        return

    @classmethod
    def get_metrics(cls, path):
        df = read_csv(path)
        yt = df['true']
        yp = df['pred']
        return eval_metrics(yt, yp, cls.threshold)


    @classmethod
    def calc(cls, model_n, version, seeds):
        dsets = ['ros', 'vm', 'd7d']
        selects = ['ba_train', 'ba_other', 'bacp_train', 'bacp_other',
                'roc_train', 'roc_other']
        #
        if isinstance(seeds, str):
            seeds = [int(x_) for x_ in seeds.split(',')]
        #
        cumul = {s_: [] for s_ in selects}
        for seed in seeds:
            cumul_seed = {s_: [] for s_ in selects}
            root = f'v05/m{model_n}/m{model_n}_{version}_seed{seed}/'
            for dt in dsets:    # dataset train
                # oof
                path = root + f'train_{dt}/{dt}_oof/softmax_m{model_n}.csv'
                res = cls.get_metrics(path)
                cumul_seed['ba_train'].append(res['th_BA'])
                cumul_seed['roc_train'].append(res['AUROC'])
                path = root + f'train_{dt}/{dt}_oof/proba_m{model_n}.csv'
                res = cls.get_metrics(path)
                cumul_seed['bacp_train'].append(res['th_BA'])
                # other
                for dp in dsets:    # dataset predict
                    if dt == dp:
                        continue
                    # non-oof
                    path = root + f'train_{dt}/{dp}/softmax_m{model_n}.csv'
                    res = cls.get_metrics(path)
                    cumul_seed['ba_other'].append(res['th_BA'])
                    cumul_seed['roc_other'].append(res['AUROC'])
                    path = root + f'train_{dt}/{dp}/proba_m{model_n}.csv'
                    res = cls.get_metrics(path)
                    cumul_seed['bacp_other'].append(res['th_BA'])
            for select in selects:
                cumul[select].append(np.array(cumul_seed[select]).mean())
        dict_avg = {select: np.array(cumul[select]).mean()
                for select in selects}
        dict_std = {select + '_std': np.array(cumul[select]).std()
                for select in selects}
        return {**dict_avg, **dict_std}


class EvalDsetdepCisic(scr.SubCommand):
    name = 'eval-dsetdep-cisic'
    threshold = 0.5

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--model-n',
                type=int, default=21,
                help='model number, eg. 21 or 41')
        parser.add_argument('--seeds',
                default='42,43,44,45,46',
                help='rng seeds, default: %(default)s')
        parser.add_argument('--latex',
                action='store_true',
                help='latex output')
        parser.add_argument('version',
                default='cv_wt_aug',
                help='version, eg. cv_wt_aug')

    def run(self):
        p = self.p
        if p.latex:
            selects = ['ba', 'bacp',]
            dsets_other = ['kwdset', 'd7d', 'ph2',]
            print('% ' + ' & '.join(selects) + '\\\\')
            for do in dsets_other:
                res = self.calc(p.model_n, p.version, p.seeds, do)
                print(do, end=' ')
                for select in selects:
                    print(f'& {100*res[select]:.1f}', end='')
                    print(f'\\err{{{100*res[select+"_std"]:.2f}}} ', end='')
                print('\\\\')
        else:
            raise NotImplementedError('currently only --latex works')
            selects = ['ba', 'bacp', 'roc',]
            print(f'{"model":18s}BA  BAcp  ROC')
            print(f'{p.version:18s}', end='')
            for select in selects:
                print(f'{res[select]:.3f}  ', end='')
            print()
            print(f'{"error":18s}', end='')
            for select in selects:
                print(f'{res[select+"_std"]:.4f} ', end='')
            print()
        return

    @classmethod
    def get_metrics(cls, path):
        df = read_csv(path)
        yt = df['true']
        yp = df['pred']
        return eval_metrics(yt, yp, cls.threshold)


    @classmethod
    def calc(cls, model_n, version, seeds, dset_pred):
        dt = dset_train = 'kwdset'
        dp = dset_pred
        selects = ['ba', 'bacp', 'roc',]
        #
        if isinstance(seeds, str):
            seeds = [int(x_) for x_ in seeds.split(',')]
        #
        cumul = {s_: [] for s_ in selects}
        for seed in seeds:
            root = f'v05/m{model_n}/m{model_n}_{version}_seed{seed}/'
            print(f'seed={seed}', end=' ')
            if dt == dp:
                # oof
                path = root + f'train_{dt}/{dt}_oof/softmax_m{model_n}.csv'
                res = cls.get_metrics(path)
                cumul['ba'].append(res['th_BA'])
                print(f'ba={100*res["th_BA"]:.1f}')
                cumul['roc'].append(res['AUROC'])
                path = root + f'train_{dt}/{dt}_oof/proba_m{model_n}.csv'
                res = cls.get_metrics(path)
                cumul['bacp'].append(res['th_BA'])
            else:
                # other, non-oof
                path = root + f'train_{dt}/{dp}/softmax_m{model_n}.csv'
                res = cls.get_metrics(path)
                cumul['ba'].append(res['th_BA'])
                print(f'ba={100*res["th_BA"]:.1f}')
                cumul['roc'].append(res['AUROC'])
                path = root + f'train_{dt}/{dp}/proba_m{model_n}.csv'
                res = cls.get_metrics(path)
                cumul['bacp'].append(res['th_BA'])
        dict_avg = {select: np.array(cumul[select]).mean()
                for select in selects}
        dict_std = {select + '_std': np.array(cumul[select]).std()
                for select in selects}
        return {**dict_avg, **dict_std}



class PlotDatasetdep(scr.SubCommand):
    name = 'plot-datasetdep'
    version_desc = {
            'cv': 'Baseline',
            'cv_wt': 'Baseline with\nclass weighted training\n(CWT)',
            'cv_wt_aug': 'Baseline with\nCWT and\naugmentation\n(AUG)',
            'cv_wt_aug_kwdset': 'Baseline with\nCWT, AUG\nC-ISIC',
            'cv_wt_aug2_kwdset': 'Baseline with\nCWT, AUG2\nC-ISIC',
            'cv_wt_aug_sam1': 'Baseline with\nCWT, AUG and\nSAM1',
            'cv_wt_aug_sam2': 'Baseline with\nCWT, AUG and\nSAM2',
            'cv_wt_aug_sam3': 'Baseline with\nCWT, AUG and\nSAM3',
            'cv_wt_aug_wd1e-5': 'Baseline with\nCWT, AUG and\nwd=1e-5',
            'cv_wt_aug_wd1e-5_kwdset': 'Baseline with\nCWT, AUG and\nwd=1e-5\nC-ISIC',
            'cv_wt_aug_wd1e-4': 'Baseline with\nCWT, AUG and\nwd=1e-4',
            'cv_wt_wd1e-5': 'Baseline with\nCWT and\nwd=1e-5',
            'cv_wt_wd1e-5_kwdset': 'Baseline with\nCWT and\nwd=1e-5\nC-ISIC',
            'cv_wt_wd3e-5': 'Baseline with\nCWT and\nwd=3e-5',
            'cv_wt_adamw': 'Baseline with\nCWT and\nAdamW',
            }

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--seeds',
                default='42,43,44,45,46',
                help='rng seed. Default: %(default)s')
        parser.add_argument('--model-n',
                type=int, default=21,
                help='model number, eg. 21')

    def run(self):
        mpl.rcParams['xtick.labelsize'] = 9.5
        p = self.p
        if 1:
            self.plot('th_BA', all_plot=1)
            self.plot('BAcp', all_plot=0)
        else:
            self.plot('AUROC', all_plot=1)
        savefig(f'fig_datasetdep.pdf')
        show()

    def plot(self, select, all_plot):
        p = self.p
        #
        if isinstance(p.seeds, str):
            seeds = [int(x_) for x_ in p.seeds.split(',')]
        #
        dsets = ['ros', 'vm', 'd7d']
        versions = ['cv', 'cv_wt', 'cv_wt_aug', 'cv_wt_aug_kwdset']
        #versions = ['cv', 'cv_wt', 'cv_wt_wd1e-5', 'cv_wt_wd1e-5_kwdset' ]
        #versions = ['cv', 'cv_wt', 'cv_wt_wd1e-5', 'cv_wt_wd1e-5_kwdset', 'cv_wt_aug_wd1e-5_kwdset', 'cv_wt_aug_kwdset' ]
        #versions = ['cv_wt_wd1e-5_kwdset', 'cv_wt_aug_wd1e-5_kwdset', 'cv_wt_aug_kwdset', 'cv_wt_aug2_kwdset' ]
        #versions = ['cv_wt_wd1e-5_kwdset', 'cv_wt_aug_kwdset']
        if select in ['th_BA', 'AUROC']:
            source = 'softmax'
        else:
            source = 'proba'
        if select == 'BAcp':
            select = 'th_BA'
        mfc = 'black' if all_plot else 'white'
        threshold = 0.5
        eps = 0.7/10
        bottom = 0.50
        text_kw = dict(
                rotation = 'vertical',
                fontfamily = 'sans-serif',
                fontsize = 6,
                fontweight = 'bold',
                )
        for vi, version in enumerate(versions):
            print(version)
            i = 0
            x_cum = []
            y_cum = []
            if version.endswith('_kwdset'):
                dsets_train = ['kwdset']
                dsets_pred = ['d7d', 'ph2', 'semmelw']
                version = version[:-7]
            else:
                dsets_train = dsets_pred = dsets
            for dt in dsets_train:
                # oof
                cumul = []
                for seed in seeds:
                    path = f'data/m{p.model_n}/m{p.model_n}_{version}_seed{seed}/train_{dt}/{dt}_oof/{source}_m{p.model_n}.csv'
                    df = read_csv(path)
                    yt = df['true']
                    yp = df['pred']
                    res = eval_metrics(yt, yp, threshold)
                    cumul.append(res[select])
                x = vi + eps * i
                y = np.array(cumul).mean()
                print(f'  {dt:3s} / {dt:3s} {y:.4f}')
                if all_plot:
                    bar(x, y-bottom, width=eps*0.9, bottom=bottom, color='red')
                if vi == 0:
                    text(x - 0.30*eps, 0.85, f'{dt} / {dt}'.upper(), **text_kw)
                elif dt == 'kwdset':
                    text(x - 0.30*eps, 0.60, f'C-ISIC / C-ISIC'.upper(), 
                            color='white', **text_kw)
                x_cum.append(x)
                y_cum.append(y)
                i += 1
            print(f'    train_dset:  {np.array(y_cum).mean():.4f}')
            #i += 1
            x_cum2 = []
            y_cum2 = []
            for dt in dsets_train:
                for dp in dsets_pred:
                    if dt == dp:
                        continue
                    # non-oof
                    cumul = []
                    if dp != 'semmelw':
                        for seed in seeds:
                            path = f'data/m{p.model_n}/m{p.model_n}_{version}_seed{seed}/train_{dt}/{dp}/{source}_m{p.model_n}.csv'
                            df = read_csv(path)
                            yt = df['true']
                            yp = df['pred']
                            res = eval_metrics(yt, yp, threshold)
                            cumul.append(res[select])
                    else:
                        # Semmelweis data is not public
                        cumul = [.815] if source == 'softmax' else [.823]
                    x = vi + eps * i
                    y = np.array(cumul).mean()
                    print(f'  {dt:3s} / {dp:3s} {y:.4f}')
                    if all_plot:
                        bar(x, y-bottom, width=eps*0.9, bottom=bottom,
                                color='#6666ff')
                    if vi == 0:
                        text(x - 0.30*eps, 0.85, f'{dt} / {dp}'.upper(),
                                **text_kw)
                    elif dt == 'kwdset':
                        text(x - 0.30*eps, 0.60, f'C-ISIC / {dp}'.upper(),
                                color='white',
                                **text_kw)
                    x_cum2.append(x)
                    y_cum2.append(y)
                    i += 1
            print(f'    other_dset:  {np.array(y_cum2).mean():.4f}')
            plot([np.array(x_cum).mean(), np.array(x_cum2).mean()],
                    [np.array(y_cum).mean(), np.array(y_cum2).mean()],
                    'ko-', markerfacecolor=mfc)

        ticks = [self.version_desc[v_] for v_ in versions]
        #
        xx = 8/2*eps + np.arange(len(ticks))
        xx[-1] -= 6/2*eps
        xticks(xx, ticks)
        ylabel('Balanced Accuracy')
        xlim(-2*eps)
        ylim(bottom, 0.95)
        gca().set_position([.11, .16, .8, .8])
        if not all_plot:
            plot([0],[0], 'ko-', markerfacecolor='black', label='averages for softmax based bal.acc.')
            plot([0],[0], 'ko-', markerfacecolor='white', label='averages for cal.prob. based bal.acc.')
            bar([0], [0], color='red', label='evaluate on train dataset')
            bar([0], [0], color='blue', label='evaluate on other dataset')
            legend(loc='upper center', bbox_to_anchor=(0.55, 0.98))



class PrintDatasetdep(scr.SubCommand):
    name = 'print-datasetdep'
    version_desc = {
            'cv': 'Baseline',
            'cv_wt': 'Baseline + class weighted training (CWT)',
            'cv_wt_aug': 'Baseline + CWT + augmentation (AUG)',
            #'cv_wt_aug_sam1': 'Baseline + CWT + AUG + SAM(no amp, half batch)',
            #'cv_wt_aug_sam3': 'Baseline + CWT + AUG + SAM(amp)',
            #'cv_wt_aug_sam2': 'Baseline + CWT + AUG + SAM(amp, tuned params)',
            #'cv_wt_aug_wd1e-5': 'Baseline + CWT + AUG + weight_decay=1e-5',
            #'cv_wt_aug_wd1e-4': 'Baseline + CWT + AUG + weight_decay=1e-4',
            #'cv_wt_wd1e-5': 'Baseline + CWT + weight_decay=1e-5',
            #'cv_wt_wd3e-5': 'Baseline + CWT + weight_decay=3e-5',
            #'cv_wt_adamw': 'Baseline + CWT with AdamW weight decay',
            }
    version_desc_v04 = {
            'cv': 'Baseline',
            'cv_wt': 'Baseline + class weighted training (CWT)',
            'cv_wt_aug': 'Baseline + CWT + augmentation (AUG)',
            #'cv_wt_wd1e-3': 'Baseline + CWT + weight_decay=1e-3',
            #'cv_wt_wd1e-5': 'Baseline + CWT + weight_decay=1e-5',
            #'cv_wt_aug_wd1e-3': 'Baseline + CWT + AUG + weight_decay=1e-3',
            #'cv_wt_aug_adamw':  'Baseline + CWT + AUG + AdamW',
            }

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--seed',
                type=int, default=42,
                help='rng seed. Default: %(default)d')
        parser.add_argument('--model-n',
                type=int, default=21,
                help='model number, eg. 21')
        parser.add_argument('--v04',
                action='store_true',
                help='v04 results. Default: v05')

    def run(self):
        p = self.p
        if p.v04:
            self.version_desc = self.version_desc_v04
        versions = self.version_desc.keys()
        print(f'{"model":18s}BA_tra BA_oth  BAcp_t BAcp_o  ROC_t  ROC_o')
        for vi, version in enumerate(versions):
            print(f'{version:18s}', end='')
            for select in ['th_BA', 'BAcp', 'AUROC']:
                res = self.calc(version, select)
                print(f'{res[0]:.4f} {res[1]:.4f}  ', end='')
            print(self.version_desc[version][:17])

    def calc(self, version, select):
        p = self.p
        dsets = ['ros', 'vm', 'd7d']
        threshold = 0.5
        #
        if select in ['th_BA', 'AUROC']:
            source = 'softmax'
        else:
            source = 'proba'
        if select == 'BAcp':
            select = 'th_BA'
        i = 0
        x_cum = []
        y_cum = []
        if 0 and p.v04:
            root = f'v04/bl/bl_{version}_seed{p.seed}/'
        else:
            root = f'data/m{p.model_n}/m{p.model_n}_{version}_seed{p.seed}/'
        for dt in dsets:
            # oof
            path = root + f'train_{dt}/{dt}_oof/{source}_m{p.model_n}.csv'
            df = read_csv(path)
            yt = df['true']
            yp = df['pred']
            dat = eval_metrics(yt, yp, threshold)
            y = dat[select]
            #print(f'  {dt:3s} / {dt:3s} {y:.4f}')
            y_cum.append(y)
            i += 1
        #print(f'    train_dset:  {np.array(y_cum).mean():.4f}')
        #i += 1
        x_cum2 = []
        y_cum2 = []
        for dt in dsets:
            for dp in dsets:
                if dt == dp:
                    continue
                # non-oof
                path = root + f'train_{dt}/{dp}/{source}_m{p.model_n}.csv'
                df = read_csv(path)
                yt = df['true']
                yp = df['pred']
                dat = eval_metrics(yt, yp, threshold)
                y = dat[select]
                #print(f'  {dt:3s} / {dp:3s} {y:.4f}')
                y_cum2.append(y)
                i += 1
        #print(f'    other_dset:  {np.array(y_cum2).mean():.4f}')
        return np.array(y_cum).mean(), np.array(y_cum2).mean()




class AreaHistogram(scr.SubCommand):
    name = 'area-histogram'

    conf = {
            'ROS':  ('v05/dsd_ros.csv', 'v05/masks/ham10000',),
            'ROS_inset':  ('v05/dsd_ros.csv', 'v05/masks/ham10000',),
            'VM':   ('v05/dsd_vm.csv',  'v05/masks/ham10000',),
            'VM_inset':   ('v05/dsd_vm.csv',  'v05/masks_rescaled/ham10000',),
            'DERM7D':  ('v05/dsd_d7d.csv', 'v05/masks/derm7pt',),
            'DERM7D_inset':  ('v05/dsd_d7d.csv', 'v05/masks_removedark2_rescaled/derm7pt',),
            }

    def run(self):
        mpl.rcParams['axes.labelsize'] = 13
        mpl.rcParams['xtick.labelsize'] = 13
        mpl.rcParams['ytick.labelsize'] = 13
        figure()
        bins = 200
        plot_bins = 40
        c_ros = self.get_cdf('ROS', bins)
        # main graph
        for dset, (dsd_pathname, data_rootname) in self.conf.items():
            if dset.endswith('_inset'):
                continue
            dat = AreaHistogram_calc(dsd_pathname, data_rootname)
            if 0 and dset != 'ROS':
                c_dset = self.get_cdf(dset, bins)
                dat_rescaled = []
                for x in dat:
                    # get cdf(x), then cdf_ros^inv(cdf(x))
                    cdf_x = c_dset[1][np.nonzero(c_dset[0] > x)[0][0]] / bins
                    cdf_x = np.clip(cdf_x, 0, 1)
                    i_cdf2 = np.nonzero(c_ros[1] > cdf_x*bins)[0][0]
                    xx = c_ros[0][i_cdf2]
                    print(x, cdf_x, xx)
                    dat_rescaled.append(xx)
                dat = dat_rescaled
            dat = np.array(dat)
            if 1:
                hist(dat, range=[0,1], bins=plot_bins, histtype='step',
                        label=dset, density=True)
            else:
                histo, edg = np.histogram(dat, range=[0,1], bins=bins,
                        density=True)
                val = np.cumsum(histo)
                xx = [edg[i_//2] for i_ in range(len(edg)*2)]
                yy = [0] + [val[i_//2] for i_ in range(len(val)*2)] + [bins]
                plot(xx, np.array(yy)/bins, label=dset)
        legend(loc='upper center', bbox_to_anchor=(0.27, 0.98))
        xlabel('lesion area / image area')
        ylabel('normalized histogram')
        # inset
        ip = inset_locator.InsetPosition(gca(), [0.48,0.48,0.5,0.5])
        axes([0,0,1,1]).set_axes_locator(ip)
        for dset, (dsd_pathname, data_rootname) in self.conf.items():
            if not dset.endswith('_inset'):
                continue
            dat = AreaHistogram_calc(dsd_pathname, data_rootname)
            dat = np.array(dat)
            hist(dat, range=[0,1], bins=plot_bins, histtype='step',
                    label=dset, density=True)
        text_kw = dict(
                fontfamily = 'sans-serif',
                fontsize = 13,
                )
        text(0.4, 2, 'after\nimage\ntransformation', **text_kw)
        #
        show()

    @classmethod
    def get_cdf(cls, dset, bins):
        dsd_pathname, data_rootname = cls.conf[dset]
        dat = AreaHistogram_calc(dsd_pathname, data_rootname)
        histo, edg = np.histogram(dat, range=[0,1], bins=bins, density=True)
        val = np.cumsum(histo)
        xx = [edg[i_//2] for i_ in range(len(edg)*2)]
        yy = [0] + [val[i_//2] for i_ in range(len(val)*2)] + [bins+1]
        return xx, yy


@joblib_memory.cache
def AreaHistogram_calc(dsd_pathname, data_rootname):
    print('called calc', dsd_pathname, data_rootname)
    dsd = read_csv(dsd_pathname)
    paths = dsd['path']
    fraction_list = []
    for pathname in paths:
        path = (Path(data_rootname) / pathname).with_suffix('.png')
        mask = imageio.imread(path).astype(bool)
        fraction_list.append(mask.sum() / (mask.shape[0] * mask.shape[1]))
    return fraction_list


class ColorHistogram(scr.SubCommand):
    name = 'color-histogram'

    conf = {
            'ros':  ('v05/dsd_ros.csv', 'data/ham10000'),
            'ros_inset':  ('v05/dsd_ros.csv', 'data/ham10000'),
            'vm':   ('v05/dsd_vm.csv',  'data/ham10000'),
            'vm_inset':   ('v05/dsd_vm.csv',  'v05/data_rescaled_adjcolor/ham10000'),
            'd7d':  ('v05/dsd_d7d.csv', 'data/derm7pt'),
            'd7d_inset':  ('v05/dsd_d7d.csv', 'v05/data_removedark2_rescaled_adjcolor/derm7pt'),
            }

    def run(self):
        mpl.rcParams['axes.labelsize'] = 13
        mpl.rcParams['xtick.labelsize'] = 13
        mpl.rcParams['ytick.labelsize'] = 13
        for i in range(3):
            figure()
            # main
            #print(gca().get_position())
            gca().set_position([0.175, 0.11, 0.775, 0.77])
            for dset, (dsd_pathname, data_rootname) in self.conf.items():
                if dset.endswith('_inset'):
                    continue
                dat = ColorHistogram_calc(dsd_pathname, data_rootname)
                edg = dat[i+3]
                val = dat[i]
                xx = [edg[i_//2] for i_ in range(len(edg)*2)]
                yy = [0] + [val[i_//2] for i_ in range(len(val)*2)] + [0]
                plot(xx, yy, label={'d7d': 'DERM7D'}.get(dset, dset.upper()))
            legend(loc='upper center', bbox_to_anchor={
                0: (0.50, 0.40),
                1: (0.75, 0.40),
                2: (0.27, 0.40),
                }[i])
            xlabel(['hue', 'saturation', 'value (intensity)'][i])
            ylabel('normalized histogram')
            if i == 0:
                xticks(np.linspace(0, 180, 7),
                        [f'{x_:g}' for x_ in np.linspace(0, 180, 7)])
            # inset
            ip = inset_locator.InsetPosition(gca(), {
                0: [0.35,0.53,0.45,0.45],
                1: [0.48,0.53,0.45,0.45],
                2: [0.15,0.53,0.45,0.45],
                }[i])
            axes([0,0,1,1]).set_axes_locator(ip)
            for dset, (dsd_pathname, data_rootname) in self.conf.items():
                if not dset.endswith('_inset'):
                    continue
                dat = ColorHistogram_calc(dsd_pathname, data_rootname)
                edg = dat[i+3]
                val = dat[i]
                xx = [edg[i_//2] for i_ in range(len(edg)*2)]
                yy = [0] + [val[i_//2] for i_ in range(len(val)*2)] + [0]
                plot(xx, yy, label={'d7d': 'DERM7D'}.get(dset, dset.upper()))
            text_kw = dict(
                    fontfamily = 'sans-serif',
                    fontsize = 13,
                    )
            if i == 0:
                xticks(np.linspace(0, 180, 4),
                        [f'{x_:g}' for x_ in np.linspace(0, 180, 4)])
            text(*({
                0: (30, 0.02),
                1: (90, 0.01),
                2: (0, 0.01),
                }[i]), 'after\nimage\ntransformation', **text_kw)
        show()

@joblib_memory.cache
def ColorHistogram_calc(dsd_pathname, data_rootname):
    dsd = read_csv(dsd_pathname)
    paths = dsd['path']
    for pathname in paths:
        path = Path(data_rootname) / pathname
        img = imageio.imread(path)
        #hsv = skimage.color.rgb2hsv(img)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        if 0 and 'd7d' in dsd_pathname:
            hue = hsv[:,:,0]
            hue -= 14
            hue[hue > 179] -= (255 - 179)
            hsv[:,:,0] = hue
            #
            sat = hsv[:,:,1]
            sat = np.interp(sat, [0, 15, 100, 255], [0, 30, 100, 255])
            sat = np.round(sat).astype(np.uint8)
            hsv[:,:,1] = sat
            #
            val = hsv[:,:,2]
            calib = np.array([
                    [0, 0],
                    [60, 80],
                    [80, 100],
                    [100, 120],
                    [120, 130],
                    [140, 143],
                    [160, 153],
                    [180, 163],
                    [200, 173],
                    [220, 189],
                    [230, 198.3],
                    [240, 211],
                    [250, 224],
                    [255, 255],
                    ])
            val = np.interp(val, calib[:,0], calib[:,1])
            val = np.round(val).astype(np.uint8)
            hsv[:,:,2] = val

        if 0 and 'vm' in dsd_pathname:
            hue = hsv[:,:,0]
            hue += 8
            hue[hue > 179] -= 180
            hsv[:,:,0] = hue
            #
            sat = hsv[:,:,1]
            sat = np.interp(sat, [0, 12, 160, 255], [0, 16, 164, 255])
            sat = np.round(sat).astype(np.uint8)
            hsv[:,:,1] = sat
            #
            val = hsv[:,:,2]
            calib = np.array([
                    [0, 0],
                    [140, 140],
                    [175, 180],
                    [195, 200],
                    [218, 220],
                    [255, 255],
                    ])
            val = np.interp(val, calib[:,0], calib[:,1])
            val = np.round(val).astype(np.uint8)
            hsv[:,:,2] = val

        hist0, edg0 = np.histogram(
                hsv[:,:,0], range=[0,180], bins=60, density=True)
        hist1, edg1 = np.histogram(
                hsv[:,:,1], range=[0,255], bins=64, density=True)
        hist2, edg2 = np.histogram(
                hsv[:,:,2], range=[0,255], bins=64, density=True)
        try:
            hist0_sum += hist0
            hist1_sum += hist1
            hist2_sum += hist2
            count += 1
        except NameError:
            hist0_sum = hist0
            hist1_sum = hist1
            hist2_sum = hist2
            count = 1
    return hist0_sum / count, hist1_sum / count, hist2_sum / count, edg0, edg1, edg2



@scr.skip
class EpsDep(scr.SubCommand):
    name = 'eps-dep'

    def run(self):
        path = 'v04/cm29/eps_dep_run0-full.txt'
        names = {
                'isic': 'C-ISIC',
                'derm7d': 'DERM7D',
                'ph2': 'PH2',
                'sed': 'Semmelweis',
                }
        datasets = list(names.keys())
        df = pd.read_csv(path, delim_whitespace=True,
                names=['eps', 'isic', 'derm7d', 'ph2', 'sed'])
        print(df.head())
        for i, dset in enumerate(datasets):
            plot(df['eps'], df[dset], 'o:', color=colors[i])
        #
        res = {
                'isic': [],
                'derm7d': [],
                'ph2': [],
                'sed': [],
                }
        for run in range(3):
            path = f'v04/cm29/eps_dep_run{run}.txt'
            df = pd.read_csv(path, delim_whitespace=True,
                    names=['eps', 'isic', 'derm7d', 'ph2', 'sed'])
            for dset in datasets:
                res[dset].append(df[dset])
        for i, dset in enumerate(datasets):
            res[dset] = np.stack(res[dset], axis=-1)
            print(dset, res[dset].std(axis=-1))
            errorbar(df['eps'], res[dset].mean(axis=-1),
                    yerr=res[dset].std(axis=-1),
                    fmt='o-', color=colors[i], label=names[dset])
        #
        path = 'v04/cm29/eps_dep_calp0.txt'
        df = pd.read_csv(path, delim_whitespace=True,
                names=['eps', 'isic', 'derm7d', 'ph2', 'sed'])
        for i, dset in enumerate(datasets):
            plot(df['eps'], df[dset], 'o-', mfc='white', color=colors[i])
            pass
        #
        legend(loc='upper right')
        xlabel('noise amplitude $\\epsilon$')
        ylabel('balanced accuracy')
        savefig('eps-dep.png', dpi=200)
        show()


@scr.skip
class TestDSD(scr.SubCommand):
    name = 'test-dsd'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--dsd',
                help='dataset descriptor csv file path')

    def run(self):
        dsd = read_csv(self.p.dsd)
        n_folds = (dsd['fold']).max() + 1
        cats = sorted(list(set(dsd['diagnosis'])))
        print('fold ' + ' '.join([f'{c_:>5s}' for c_ in cats + ['TOTAL']]))
        for fold in range(n_folds):
            print(f'{fold:4d}', end=' ')
            for cat in cats:
                print(f'{((dsd["diagnosis"] == cat) & (dsd["fold"] == fold)).sum():5d}', end=' ')
            print(f'{(dsd["fold"] == fold).sum():5d}')


main = scr.main_multi(__name__)

if __name__ == '__main__':
    main()

# vim: set sw=4 sts=4 expandtab :
