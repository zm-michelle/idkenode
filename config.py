import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variables for model training")
    
    # Dont forget to add tqdm 
    # sh script
    parser.add_argument("--nrounds", default=22, type=float, help="main train loopp rounds")
    parser.add_argument("--eta_policy", default=1e-3, type=float, help="Learning_rate policy")
    parser.add_argument("--eta_dynamics", default=1e-3, type=float, help="Learning_rate dynamics")

    parser.add_argument("--noise", default=0.0, type=float, help="Observation noise std")

    parser.add_argument("--dt", default=0.1, type=float, help="mean-time difference between observations")
    
    parser.add_argument("--ts_grid", default="fixed", type=str, help="the distribution for the observation time differences: ['fixed','uniform','exp']")
    parser.add_argument("--envs_cls", default="CTPendulum", type=str, help="[CTPendulum, CTCartpole, CTAcrobot]")
    parser.add_argument("--n_ens", default=10, type=int, help="ensemble size")
    parser.add_argument("--nl_f", default=3, type=int, help="number of hidden layers in the differential function")
    parser.add_argument("--nn_f", default=200, type=int, help="number of hidden neurons in each hidden layer of f")
    parser.add_argument("--act_f", default="elu", type=str, help="activation of f (should be smooth)")
    parser.add_argument("--dropout_f", default=0.05, type=float, help="dropout parameter (needed only for deep pilco)")
    parser.add_argument("--learn_sigma", default=False, type=bool, help="whether to learn the observation noise or keep it fixed")
    parser.add_argument("--nl_g", default=2, type=int, help="number of hidden layers in the policy function")
    parser.add_argument("--nn_g", default=200, type=int, help="number of hidden neurons in each hidden layer of g")
    parser.add_argument("--act_g", default="relu", type=str, help="activation of g")
    parser.add_argument("--nl_V", default=2, type=int, help="number of hidden layers in the state-value function")
    parser.add_argument("--nn_V", default=200, type=int, help="number of hidden neurons in each hidden layer of g")
    parser.add_argument("--act_V", default="tanh", type=str, help="activation of V (should be smooth)")
   
    parser_params = parser.parse_args()

    #  Parameters based on the DIAYN and SAC papers.
    # region default parameters
    
    # endregion
    total_params = {**vars(parser_params) }
    return total_params