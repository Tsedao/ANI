import matplotlib.pyplot as plt
import numpy as np

def plot_intensity(dataset, model, save_path=None):
    try:        
        data_event = dataset.events
    except AttributeError:
        data_event = None 
    horizon = dataset.length
    num_nodes = dataset.num_nodes

    fig, axes = plt.subplots(
                        num_nodes*2,1,
                        figsize=(5,6),
                        gridspec_kw = {'height_ratios':sum([[3,1] for i in range(num_nodes)],[])},
                        sharex='col'
                        )
    
    for i in range(1):
        # Neural process returns distribution over y_target
        model.ode_solver_name = 'rk4'
        (_,time, _), dyna_actions_dict, intensities_nocontrol = model(
            event_data=None,
            event_time=None, 
            t0=0,
            t1=horizon,
            action_mask=None,
            inference=True,
            policy_learning=False
        )

        for j in range(num_nodes):
            axes[j*2].plot(
                    time.cpu().numpy(), 
                    intensities_nocontrol.cpu().numpy()[0,:,j], 
                    alpha=1.0, c='b'
                )
            axes[j*2].set_ylabel('$\lambda(t)_{%d}$' % j, fontsize=7)
            axes[j*2].tick_params(bottom=False)
            # axes[j].set_ylim([-0.5,1.5])

    if data_event is not None:
        for j in range(num_nodes):
            # axes[j*2+1].scatter(x_context[0].numpy(), y_context[0,:,j].numpy(), c='k',alpha=0.1)

            # plot events
            subseq = data_event[data_event[:,1]==j][:,0]
            axes[2*j+1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'ko', alpha=0.2)
            axes[2*j+1].yaxis.set_visible(False)

            axes[2*j+1].set_xlim([0, 100])
            axes[2*j+1].tick_params(bottom=False)
    if save_path is not None:
        fig.savefig(save_path)