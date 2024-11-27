keys = ["generator", "m_p_feat", "m_p_edge_drop", "m_p_edge_add", "er_min_num_vertices", "er_max_num_vertices", "er_p_edge", "ct_min_num_components", 
        "ct_max_num_components", "ct_p_cycle", "ct_cycle_lens", "ct_max_chordless_cycle_len"]

def add_generator_args(parser):
    parser.add_argument('--generator', type=str, default='m',
                        help="Graph generator (values: m, ct, er)")
    
    # Mutation graph generator
    parser.add_argument('--m_p_feat', type=float, default=0.05,
                        help="(m) Probability that a single feature gets mutated (default: 0.05)")  
    parser.add_argument('--m_p_edge_drop', type=float, default=0.05,
                    help="(m)  Probability that an edge gets dropped (default: 0.05)")   
    parser.add_argument('--m_p_edge_add', type=float, default=0.05,
                    help="(m) Probability that an edge gets added (default: 0.05)")  
    
    # Erdos-Renyi graph generator
    parser.add_argument('--er_min_num_vertices', type=int, default=10,
                    help='(er) Min number of vertices in graph')
    parser.add_argument('--er_max_num_vertices', type=int, default=18,
                    help='(er) Maximum number of vertices in a graph')
    parser.add_argument('--er_p_edge', type=float, default=0.2,
                    help="(er) Probability that two vertices are connected by an edge")  
    
    # Cycle tree graph generator
    parser.add_argument('--ct_min_num_components', type=int, default=6,
                    help='(ct) Minimum number of components (biconnected components + vertices in trees)')
    parser.add_argument('--ct_max_num_components', type=int, default=17,
                    help='(ct) Maximum number of components (biconnected components + vertices in trees)')
    parser.add_argument('--ct_p_cycle', type=float, default=0.2,
                    help="(ct) Probability that a component will be turned into a biconnected component")  
    parser.add_argument('--ct_cycle_lens', type=str, default="[4, 5, 6, 8, 9, 10]",
                        help="(ct) Possible biconnected component sizes")
    parser.add_argument('--ct_max_chordless_cycle_len', type=int, default=6,
                    help='(ct) Any biconnected component bigger than this will have a chord)')