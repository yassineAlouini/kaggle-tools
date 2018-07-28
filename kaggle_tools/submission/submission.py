import datetime
import os

import pandas as pd


def create_submission(predictions, prediction_columns, submission_folder, info):
    result = pd.DataFrame(data=predictions, columns=prediction_columns)
    now = datetime.datetime.now()
    if not os.path.isdir(submission_folder):
        os.mkdir(submission_folder)
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join(submission_folder, 'submission_' + suffix + '.csv')
    result.to_csv(sub_file, index=False)
    print("Finished creating the submission!")
