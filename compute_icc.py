import pandas as pd
import numpy as np
from pingouin import intraclass_corr
from matplotlib import pyplot as plt

def auto_detect_and_calculate_icc(csv_path, roi_column='ROI', series_column='SeriesDescription' ,feature_column='deepfeatures'):
    data = pd.read_csv(csv_path)
    data['SeriesDescription'] = data['SeriesDescription'].apply(lambda x: x.split('_')[0])
    
    if feature_column in data.columns and data[feature_column].dtype == 'object':
        data[feature_column] = data[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        max_len = data[feature_column].apply(len).max()
        feature_df = pd.DataFrame(data[feature_column].tolist(), index=data.index)
        feature_df.columns = [f"feature_{i}" for i in range(max_len)]
        data = pd.concat([data.drop(columns=[feature_column]), feature_df], axis=1)
    
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in [roi_column, series_column]]
    
    results = []
    for feature in feature_columns:
        # targets = np.ones_like(roi_column)
        # for item in series_column.unique():
        #     targets[series_column==item] = np.mean(roi_column[series_column==item])
        icc_data = data[[series_column, roi_column, feature]].dropna()
        # Add random noise to the feature column to avoid duplicate values
        # icc_data[feature] = icc_data[feature] + np.random.normal(0, 1, len(icc_data))
        icc_data.columns = ['raters', 'targets', 'ratings']
        try:
            icc = intraclass_corr(data=icc_data, raters='raters', targets='targets', ratings='ratings').set_index('Type').at['ICC3', 'ICC']
            print(icc)
            if icc < -3:
                icc = intraclass_corr(data=icc_data, raters='raters', targets='targets', ratings='ratings').set_index('Type').at['ICC3', 'ICC']
            if icc < -9:
                plt.hist(icc_data['ratings'])
                plt.title(f"{feature} - ICC: {icc}")
                plt.show()
                x = np.array(list(range(len(icc_data['ratings']))))
                # Box plots per rater
                for rater in icc_data['raters'].unique():
                    plt.boxplot([icc_data['ratings'][icc_data['raters'] == rater] for rater in icc_data['raters'].unique()], showfliers=False)
                plt.show()
            results.append({'Feature': feature, 'ICC': icc})
        except Exception as e:
            results.append({'Feature': feature, 'ICC': np.nan})

    return pd.DataFrame(results)


def main():
    csv_path = './features_swinunetr_full.csv'
    # csv_path = './pyradiomics_features_full.csv'
    # csv_path = './features_ocar_full.csv'
    icc_results = auto_detect_and_calculate_icc(csv_path)
    icc_results_sorted = icc_results.sort_values(by='ICC', ascending=False)
    print(icc_results_sorted)
    # iccs = []
    # for csv_path(['./features_swinunetr_full.csv', './pyradiomics_features_full.csv', './features_ocar_full.csv']):
    #     icc_results = auto_detect_and_calculate_icc(csv_path)
    #     iccs.append(icc_results)
    # # Box plot of items in icc_results['ICC'] and drop nan:
    plt.boxplot(icc_results['ICC'].dropna())
    plt.show()
    plt.boxplot(icc_results['ICC'].dropna(), showfliers=False)
    plt.show()
    # Box plot the items in ICCs and call them pyradiomics, ccn, swinunetr:
    #plt.boxplot([icc_results['ICC'].dropna() for icc_results in iccs], showfliers=False)

    output_path = './features_swinunetr_full_icc_results.csv'
    # output_path = './features_pyradiomics_full_icc_results.csv'
    # output_path = './features_ocar_full_icc_results.csv'
    icc_results_sorted.to_csv(output_path, index=False)
    print(f"Les résultats ICC sont enregistrés dans {output_path}")

if __name__ == '__main__':
    main()