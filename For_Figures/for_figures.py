import matplotlib.pyplot as plt

def call_pltsettings(scale_dpi = 1,scale = 1,fontscale = 1.5,ratio = [1,1]):
    plt.style.use('seaborn-talk')
#    plt.rcParams['font.family'] = 'serif' 
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 11*scale*fontscale
    plt.rcParams['axes.labelsize'] = 11*scale*fontscale
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 8*scale*fontscale
    plt.rcParams['ytick.labelsize'] = 8*scale*fontscale
    plt.rcParams['legend.fontsize'] = 9*scale*fontscale
    plt.rcParams['figure.titlesize'] = 12*scale*fontscale
    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.5
     
    para_dict = {
            'lines.linewidth': 1.2*scale,
            'figure.figsize' : [6.4*scale*ratio[0],4.8*scale*ratio[1]],
            'axes.titlesize' : 9*scale,
            'axes.titlepad'  : 3.0*scale,
            'axes.xmargin'   : 0.01*scale,
            'axes.ymargin'   : 0.01*scale,
            'axes.labelpad'  : 1.5*scale,
            'xtick.major.pad': 2*scale,
            'xtick.minor.pad': 1.9*scale,
            'ytick.major.pad': 2*scale,
            'ytick.minor.pad': 1.9*scale,
            'figure.dpi'     : 100*scale_dpi,
            'figure.subplot.left'    : 0.1,  
            'figure.subplot.right'   : 0.9,    
            'figure.subplot.bottom'  : 0.1,   
            'figure.subplot.top'     : 0.88,   
            'figure.subplot.wspace'  : 0.2,                                     
            'figure.subplot.hspace'  : 0.2,
            'figure.autolayout'      : True,
            'figure.constrained_layout.use': False,
            'path.simplify_threshold': 1,
            'savefig.dpi'            : 100 

                 }
    for key,value in para_dict.items():
        plt.rcParams[key] = value

def pltend():
    plt.grid(color = 'gray',linestyle='-',lw = 1)
    plt.grid(b=True, which='minor', color='gray', linestyle='--',lw = 0.2)
    plt.show()