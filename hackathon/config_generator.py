import pandas as pd
import random
import numpy as np
from anarci import anarci


# hydrophobic/aromatic enrichment typical in binding interfaces
BINDING_PROPENSITY = {
    'Y': 1.0, 'W': 0.9, 'F': 0.9, 'H': 0.8, 'R': 0.7, 'K': 0.7, 'D': 0.5, 'E': 0.5,
    'S': 0.4, 'T': 0.4, 'N': 0.4, 'Q': 0.4, 'A': 0.3, 'L': 0.3, 'I': 0.3,
    'V': 0.3, 'G': 0.2, 'C': 0.2, 'P': 0.1, 'M': 0.3
}

CDR_WEIGHTS = {(27, 38): 0.8, (56, 65): 1.0, (105, 117): 1.3}

CDR_RANGES = {
    "H": [(27, 38), (56, 65), (105, 117)],  # heavy
    "L": [(27, 38), (56, 65), (105, 117)]   # light (same numbering convention)
}

def identify_antibody_hotspots(numbering, heavy_seq, light_seq, top_n):
    """
    Predict likely binding hotspot residues and return a list of lists like:
    [['H', pos], ['L', pos], ...]
    """
    def extract_hotspots(chain_label, numbered_domains, seq):
        if numbered_domains[0] is None:
            return []
        domain_numbering, start_index, end_index = numbered_domains[0]
        hotspots = []

        for (cdr_start, cdr_end) in CDR_RANGES[chain_label]:
            for idx, ((imgt_pos, ins), aa) in enumerate(domain_numbering):
                if aa == '-' or imgt_pos is None:
                    continue
                if cdr_start <= imgt_pos <= cdr_end:
                    rel_pos = (imgt_pos - cdr_start) / (cdr_end - cdr_start + 1)
                    center_weight = 1 - abs(rel_pos - 0.5) * 2 
                    prop = BINDING_PROPENSITY.get(aa, 0.2)
                    score = prop * (0.5 + center_weight) * CDR_WEIGHTS.get((cdr_start, cdr_end), 1.0)
                    seq_index = start_index + idx
                    hotspots.append((seq_index, aa, score))

        hotspots = sorted(hotspots, key=lambda x: x[2], reverse=True)
        return [[chain_label, pos] for pos, aa, score in hotspots[:top_n]]

    H_sites = extract_hotspots("H", numbering[0], heavy_seq)
    L_sites = extract_hotspots("L", numbering[1], light_seq)
    
    return {'H': H_sites, 'L': L_sites}

def generate_all_possible_configs(heavy_chain_seq, light_chain_seq):
    sequences = [('H', heavy_chain_seq), ('L', light_chain_seq)]
    numbering, alignment_details, hit_tables = anarci(sequences, scheme="imgt", output=False)
    hotspots = identify_antibody_hotspots(numbering, heavy_chain_seq, light_chain_seq, 20) 
    # total_list = hotspots[:8]+hotspots[:6]+hotspots[:4]
    return [hotspots['H'][:4] + hotspots['L'][:4], hotspots['H'][:3] + hotspots['L'][:3], hotspots['H'][:2] + hotspots['L'][:2]]