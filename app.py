import rapidfuzz
import numpy as np
import pathlib
import json

def run_pipeline():
    cachedir = pathlib.Path('cache')
    cachedir.mkdir(exist_ok=True)

    with open(cachedir / 'resources' / 'example_prediction.json', 'r') as f:
        propjson = json.loads(f.read())

    proptitles = [prop.get('property').get('title') for prop in propjson]
    propfile = cachedir / 'resources' / 'predicted_property_names.txt'
    propfile.write_text('\n'.join(proptitles))

    def fuzzy_match_properties(input_path,output_path):
        relevant_properties = (input_path).read_text().splitlines()
        predicted_properties = (cachedir / 'predicted_property_names.txt').read_text().splitlines()
        matched_properties = []
        for rel_prop in relevant_properties:
            scores = [rapidfuzz.fuzz.token_sort_ratio(rel_prop.lower(), pred_prop.lower()) for pred_prop in predicted_properties]
            best_match_idx = int(np.argmax(scores))
            matched_properties.append(predicted_properties[best_match_idx])
        output_path.write_text('\n'.join(matched_properties))
        return list(set(matched_properties))

    projects = ['hepatotoxicity','nephrotoxicity','developmental_neurotoxicity']

    for project in projects:
        fuzzy_match_properties(
            input_path=cachedir / 'projects' / project / 'chatgpt_selected_features.txt',
            output_path=cachedir / 'projects' / project / 'matched_properties.txt')

    import RAPtool.parse_chemicals as parse_chemicals
    for project in projects:
        parse_chemicals.parse_chemicals(
            input_path=cachedir / 'projects' / project / 'chemicals.txt',
            output_path=cachedir / 'projects' / project / 'parsed_chemicals.csv'
        )

    import RAPtool.categorize_chemicals as categorize_chemicals
    for project in projects:
        categorize_chemicals.categorize_chemicals(
            input_path=cachedir / 'projects' / project / 'parsed_chemicals.csv',
            output_path=cachedir / 'projects' / project 
        )

    import RAPtool.predict_chemicals as predict_chemicals
    for project in projects:
        predict_chemicals.predict_chemicals(
            input_path=cachedir / 'projects' / project / 'classified_chemicals.csv',
            output_path=cachedir / 'projects' / project / 'predictions.parquet'
        )

    import RAPtool.select_feature as select_feature
    for project in projects:
        outdir = cachedir / 'projects' / project / 'selected_properties'
        outdir.mkdir(exist_ok=True)
        select_feature.select_feature(
            input_path=cachedir / 'projects' / project / 'predictions.parquet',
            output_path= outdir,
            max_features=150
        )

    runtag = 'pilot'
    import RAPtool.build_heatmap as build_heatmap
    for project in projects:
        outdir = cachedir / 'projects' / project / 'heatmap_dir'
        outdir.mkdir(exist_ok=True)
        for feature_selection_method in ['lasso', 'random_forest', 'mutual_info', 'rfe']:
            build_heatmap.build_heatmap(
                input_path=cachedir / 'projects' / project / 'predictions.parquet',
                output_path=outdir / f'{feature_selection_method}_heatmap_{runtag}.png',
                feature_selection_method= feature_selection_method
            )

    feature_selection_method = 'mutual_info'
    import RAPtool.build_stripchart as build_stripchart
    for project in projects:
        outdir = cachedir / 'projects' / project / 'stripchart_dir'
        outdir.mkdir(exist_ok=True)
        for agg_func in ['median', 'mean']:
            build_stripchart.build_stripchart(
                input_path=cachedir / 'projects' / project / 'predictions.parquet',
                output_path=outdir / f"{agg_func}_{feature_selection_method}_stripchart_{runtag}.png",
                agg_func=agg_func,
                feature_selection_method= feature_selection_method
            )

if __name__ == "__main__":
    run_pipeline()
