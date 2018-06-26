import matplotlib.pyplot as plt
from math import floor


print_colors = ['b','g','r','c','m','y','k']
markers      = ['o','x']


def plot(algorithms, run_stats):
    
    # Run stats have the following information
    #(mean_error_count, mean_update_count, mean_time, np.mean(mistakes_arr, axis=1), np.mean(nb_SV_cum_arr, axis=1), np.mean(time_cum_arr, axis=1), captured_t))
    mistakes   = [x[3] for x in run_stats]
    updates    = [x[4] for x in run_stats]
    time       = [x[5] for x in run_stats]
    captured_t = [x[6] for x in run_stats]
    
    plot_w_info(captured_t, mistakes, algorithms, "Number of Samples", "Online Cumulative Mistake Rate")
    plot_w_info(captured_t, updates, algorithms, "Number of Samples", "Online Cumulative Number of Updates")
    plot_w_info(captured_t, time, algorithms, "Number of Samples", "Online Cumulative Time Cost (s)")

# This function receives the information to be plotted on x and y axis as well as the axis names and the title
def plot_w_info(x, y,algorithms, x_name, y_name):
    
    # Plot Mistake rate
    fig, ax = plt.subplots()
    for i in range(len(algorithms)):
        algorithm  = algorithms[i]
        current_x  = x[i]
        current_y  = y[i]

        ax.plot(current_x, current_y, marker=markers[floor(i/len(print_colors))], color=print_colors[i%len(print_colors)], label=algorithm)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
        
        
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
    
    plt.show()
