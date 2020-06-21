import numpy as np
from scipy.special import gamma
from scipy.misc import comb
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse


def plot_conf_ellipse(mu,cov,ax=None,color='m',crosshairs=False,linestyle='-',linewidth=1,ci=.95):
    '''
    plot ellipse given mu and cov
    '''
    if np.max(cov)>1e8:
        print('too big to plot')
        return ax
    if ax is None:
        f,ax = plt.subplots()
    if crosshairs:
        ax = plot_crosshairs(mu,cov,ax,color='k')
    vals,vecs = np.linalg.eig(cov)
    theta = (360/(2*np.pi))*np.arctan(vecs[1,0]/vecs[0,0])
    # able to change CI now.
    scale = chi2.ppf(ci,2)
    # w = np.sqrt(vals[0]*scale)*2
    # h = np.sqrt(vals[1]*scale)*2
    w = np.sqrt(vals[0]*scale)
    h = np.sqrt(vals[1]*scale)
    e = Ellipse(xy = mu.ravel() ,width =  w, height = h,angle=theta,linewidth=linewidth )

    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
#    e.set_alpha(.75)
    e.set_edgecolor((color))
    e.set_linestyle((linestyle))
    e.set_facecolor(('none'))

    # rescale axis to fit Ellipse
    scale = 3
    stdv_x = scale*np.sqrt(cov[0,0])
    stdv_y = scale*np.sqrt(cov[1,1])
    ax.set_xlim([mu[0]-stdv_x,mu[0]+stdv_x])
    ax.set_ylim([mu[1]-stdv_y,mu[1]+stdv_y])

    return ax

def make_uncertainty_plot(data=[],log_trans=False,scatter_plot=True,contour_plot=False,
                            ellipse=True,parameter_names = [], crlb = [], kde=False,
                            true_parameters=[],colors=[],ax=[],crlb_params=[],group_labels=[]):
    '''
    a;lskfdj;alsdkfj;alsjkfd
    a;sldkfja;sldkfja;sldjkf
    '''
    # add parameter names
    if len(data)>0:
        npars = data[0].shape[1]
        ngroups = len(data)
    elif len(crlb)>0:
        ngroups = len(crlb)
        npars = crlb[0].shape[0]

    if len(parameter_names)==0:
        for i in range(npars):
            parameter_names.append(r'$\lambda_{0}$'.format(i))

    if len(group_labels)==0:
        for i in range(ngroups):
            group_labels.append('Group {0}'.format(i))

    # log transform data
    if log_trans:
        data = np.log(data)
        true_parameters = np.log(true_parameters)

    # if no colors are specified, pull random colors from a colormap.
    cmap = cm.viridis.colors
    if len(colors)==0:
        colors = []
        for i in range(ngroups):
            colors.append(cmap[np.random.randint(255)])

    # make a figure and axis if there are none.
    if len(ax)==0:
        f,ax = plt.subplots(npars,npars,figsize=(npars*2*1.3,npars*2))


    for dd in range(ngroups):
        if len(data)>0:
            mu = np.mean(data[dd],axis=0)
            cov = np.cov(data[dd].T)
        for i in range(npars):
            for j in range(i+1):
                ax_now = ax[i,j]
                # plot the diagonal bits.
                if i==j:
                    if kde:
                        # # do kde to estimate histograms.
                        # # still to be written.
                        # [f,xi] = ksdensity(data(:,i));
                        # fill([xi fliplr(xi)],[f zeros(1,length(f))],options.colors);
                        # if length(options.true_parameters)>0
                        #     plot([options.true_parameters(i) options.true_parameters(i)],[0,max(f)])
                        pass
                    else:
                        if len(data)>0:
                            # plot the raw data.
                            ax_now.hist(data[dd][:,i],bins=30,color=colors[dd])
                            # if length(options.true_parameters)>0
                            #     N = histcounts(data(:,i));
                            #     plot([options.true_parameters(i) options.true_parameters(i)],[0,max(N)])
                        else:
                            pass
                else:
                    if len(data)>0:
                        if ellipse:
                            cov_now = cov[[i,j],:][:,[i,j]]
                            ax_now = plot_conf_ellipse(mu[[i,j]],cov_now,ax=ax_now,color=colors[dd],crosshairs=False,linestyle='-',linewidth=2,ci=.95)
                    if len(crlb)>0:
                        for n in range(len(crlb)):
                            if len(crlb_params)>0:
                                mu_crlb = crlb_params[dd][[j,i]]
                            else:
                                if len(data)>0:
                                    mu_crlb = np.copy(mu)
                                elif len(true_parameters)>0:
                                    mu_crlb = true_parameters[[j,i]]
                                else:
                                    mu_crlb = np.array([0,0])
                            crlb_n = crlb[n];
                            ax_now = plot_conf_ellipse(mu_crlb,crlb_n,ax=ax_now,color=colors[n],crosshairs=False,linestyle='-',linewidth=1,ci=.95)
                    if scatter_plot:
                        # this should probably check if there is data
                        if len(data)>0:
                            ax_now.scatter(data[dd][:,j],data[dd][:,i],s=10,c=np.array([colors[dd]]))

                    if contour_plot:
                        if len(data)>0:
                            [counts,C] = np.histogramdd(data[dd][:,[j,i]],[10,10]);
                            ax_now.contour(C[0],C[1],counts.T)
                        else:
                            pass

                    if len(true_parameters)>0:
                        ax_now.scatter(true_parameters[j],true_parameters[i],c = 'k',s=8,marker='D')

                if i==npars-1:
                    ax_now.set_xlabel(parameter_names[j],fontsize=15)
                if j==0:
                    ax_now.set_ylabel(parameter_names[i],fontsize=15)


    #fix up axes

    for i in range(npars):
        for j in range(npars):
            if i==j:
                # make y axis labels be on the right for all plots.
                ax[i,j].tick_params(axis='y',labelsize=8,labelcolor='gray')
                if i>0:
                    ax[i,j].yaxis.tick_right()
            if i==npars-1:
                # add xlabels and ticks to bottom row.
                ax[i,j].set_xlabel(parameter_names[j])
                ax[i,j].tick_params(axis='x',labelsize=8,labelcolor='gray')
            if j==0:
                # add yticks and ylabels
                ax[i,j].set_ylabel(parameter_names[i])
                if i>0:
                    ax[i,j].tick_params(axis='y',labelsize=8,labelcolor='gray')
            if i < (npars-1):
                    ax[i,j].set_xticks([])
                    ax[i,j].set_xticklabels([])
            if (j > 0 and j<i):
                    ax[i,j].set_yticks([])
                    ax[i,j].set_yticklabels([])
            if i<j:
                # turn off top right plots
                ax[i,j].axis('off')
                ax[i,j].set_xticklabels([])
                ax[i,j].set_yticklabels([])

    # add group labels.
    x = ax[-2,-1].get_xlim()
    ax[-2,-1].set_ylim([0,ngroups+1])
    ypos = np.arange(ngroups+1)[::-1]
    fsize=10
    for i in range(ngroups):
        ax[-2,-1].text(x[0],ypos[i],group_labels[i],color=colors[i],fontsize=fsize)

    return ax
