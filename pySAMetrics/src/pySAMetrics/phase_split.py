import numpy as np

def split_QN(QN, TABS, t0=273.15):
    """Split QN into QN_ice and QN_liq based on temperature."""
    QN_ice = np.where(TABS < t0, QN, 0)
    QN_liq = np.where(TABS >= t0, QN, 0)
    return QN_ice, QN_liq

def split_QP(QP, TABS, t0=273.15):
    """Split QP into QP_ice and QP_liq based on temperature."""
    QP_ice = np.where(TABS < t0, QP, 0)
    QP_liq = np.where(TABS >= t0, QP, 0)
    return QP_ice, QP_liq