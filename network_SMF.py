import SMF_BCD as SMF
import numpy as np

def main():
    # Get the data from run_SNLD
    X = np.loadtxt("Output/A.txt")
    Y = np.loadtxt("Output/y.txt")
    results_dict_list = []
    SMF_Train = SMF.SDL_BCD([X, Y.T])
    results_dict_new = SMF_Train.fit(iter=200, subsample_size=None,
                                                            beta = 1,
                                                            option = "filter",
                                                            search_radius_const=200*np.linalg.norm(X),
                                                            update_nuance_param=False,
                                                            if_compute_recons_error=True, if_validate=False)
    results_dict_new.update({'method': 'SDL-filt'})
    results_dict_new.update({'beta': 1})
    results_dict_new.update({'time_error': results_dict_new.get('time_error')})
    results_dict_list.append(results_dict_new.copy())
    
if __name__ == "__main__":
    main()
