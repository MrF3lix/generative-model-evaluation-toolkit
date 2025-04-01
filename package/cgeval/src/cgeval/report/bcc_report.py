import json
from prettytable import PrettyTable

from cgeval import Report

class CountReport(Report):
    def __init__(self, labels, report, samples):
        self.labels = labels
        self.report = report
        self.samples = samples
        self.images = []
    
    def __str__(self):

        #                 mean       std    median      5.0%     95.0%     n_eff     r_hat
        # mu_col_0[0]     0.76      0.06      0.76      0.65      0.86  30126.29      1.00
        # mu_col_0[1]     0.19      0.06      0.19      0.09      0.29  32938.32      1.00
        # mu_col_0[2]     0.06      0.03      0.05      0.00      0.11  44346.39      1.00
        # mu_col_1[0]     0.02      0.02      0.02      0.00      0.06  48882.46      1.00
        # mu_col_1[1]     0.92      0.04      0.93      0.87      0.99  48030.68      1.00
        # mu_col_1[2]     0.05      0.03      0.04      0.00      0.10  48911.05      1.00
        # mu_col_2[0]     0.02      0.02      0.02      0.00      0.05  47148.38      1.00
        # mu_col_2[1]     0.34      0.06      0.34      0.23      0.43  30744.42      1.00
        # mu_col_2[2]     0.64      0.06      0.64      0.54      0.74  31238.86      1.00
        #   p_true[0]     0.31      0.03      0.31      0.26      0.36  28397.70      1.00
        #   p_true[1]     0.33      0.04      0.34      0.27      0.40  31251.04      1.00
        #   p_true[2]     0.36      0.04      0.35      0.30      0.41  30846.36      1.00


        t = PrettyTable(['Label', 'Input Prevalence', 'Metric Rating Prevalence'])
        t.align["Support"] = "r"
        
        for label in self.labels:
            name = label.name
            t.add_row([
                name,
                self.report[name]['count_inputs'],
                self.report[name]['count_metric_ratings'],
            ], divider=True)

        return str(t)
    
    def toJSON(self):
        return json.dumps(
            self.report,
            sort_keys=True,
            indent=2
        )