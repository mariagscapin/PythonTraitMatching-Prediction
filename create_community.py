from create_species import *
from simulate_interaction import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer





def createCommunity(a=None, b=None, z=None, community=None, response=None, positive=None, impute=True, log=True):
    class_community = {}

    check_input(a, b, z, community, response, positive)

    imp_data = []
    z_list = []

    if a is not None:
        imp_data.append(impute_data(a, b))

    if a is None and community is not None:
        if isinstance(community, pd.DataFrame):
            class_community['data'] = community
        else:
            for elements in community:
                if isinstance(elements, pd.DataFrame):
                    class_community['data'] = elements
                else:
                    imp_data.append(impute_data(elements[0], elements[1]))
                    z_list.append(elements[2])


    if imp_data:
        inter_output = create_inter(imp_data, z, log)
        class_community['data'] = inter_output['data']
        class_community['target'] = inter_output.get('target')
        class_community['type'] = inter_output.get('type')


    if response and 'target' not in class_community:
        class_community['target'] = response
        if len(class_community['data'][response].unique()) > 2:
            class_community['type'] = "Regression"
        else:
            class_community['type'] = "Classification"


    class_community['data']['X'] = class_community['data']['X'].astype(str)
    class_community['data']['Y'] = class_community['data']['Y'].astype(str)

    class_community['class'] = ["Community", class_community['type']]

    return class_community


def create_inter(imp_data, z, log):
    output = {}
    final_data_frames = []

    for imp in imp_data:
        if len(imp) == 1:
            final_data_frames.append(imp[0])
        else:
            X = imp[0]
            Y = imp[1]

            X.columns = ['X'] + list(X.columns[1:])
            Y.columns = ['Y'] + list(Y.columns[1:])


            A_cols = [col for col in X.columns if col != 'X']
            B_cols = [col for col in Y.columns if col != 'Y']


            final_data = []
            Z_m = z.melt(var_name='X', value_name='Y')
            Z_m.dropna(inplace=True)

            for _, a_row in X.iterrows():
                for _, b_row in Y.iterrows():
                    combined_row = {'X': a_row['X'], 'Y': b_row['Y']}
                    combined_row.update({a_col: a_row[a_col] for a_col in A_cols})
                    combined_row.update({b_col: b_row[b_col] for b_col in B_cols})
                    final_data.append(combined_row)

            final_df = pd.DataFrame(final_data)

            if not final_df.empty:
                final_df['target'] = Z_m['Y']

                unique_values = final_df['target'].unique()
                output['type'] = "Classification" if len(unique_values) <= 2 else "Regression"

                if output['type'] == "Classification":
                    final_df['target'] = Z_m['Y'].replace({1.0: 'positive', 0.0: 'negative'})

                if output['type'] == "Regression" and log:
                    if 'target' in final_df.columns and pd.api.types.is_numeric_dtype(final_df['target']):
                        final_df['target'] = np.log1p(final_df['target'])

                    for col in A_cols + B_cols:
                        if col in final_df.columns and pd.api.types.is_numeric_dtype(final_df[col]):
                            final_df[col] = np.log1p(final_df[col])

                final_data_frames.append(final_df)


    output['data'] = final_data_frames[0] if final_data_frames else pd.DataFrame(columns=['X', 'Y', 'target'])
    output['target'] = final_df['target']
    return output


def impute_data(a, b):
    X = a.drop_duplicates(subset=a.columns[0])
    Y = b.drop_duplicates(subset=b.columns[0])

    imputer = IterativeImputer(max_iter=10, random_state=0)

    X_imp = imputer.fit_transform(X.iloc[:, 1:]) if X.iloc[:, 1:].isnull().sum().sum() > 0 else X.iloc[:, 1:].values
    Y_imp = imputer.fit_transform(Y.iloc[:, 1:]) if Y.iloc[:, 1:].isnull().sum().sum() > 0 else Y.iloc[:, 1:].values

    X_imp = pd.DataFrame(np.column_stack([X.iloc[:, 0], X_imp]), columns=X.columns)
    Y_imp = pd.DataFrame(np.column_stack([Y.iloc[:, 0], Y_imp]), columns=Y.columns)

    return [X_imp, Y_imp]


def check_input(a=None, b=None, z=None, community=None, response=None, positive=None):
    if all(x is None for x in [a, b, z, community]):
        raise ValueError("Provide groups and their interaction matrix or a community list")

    if a is None and community is not None:
        if isinstance(community, pd.DataFrame) and response is None:
            raise ValueError("Please provide response for the community")
        elif isinstance(community, list):
            for elements in community:
                if isinstance(elements, pd.DataFrame):
                    check_input(community=elements, response=response)
                else:
                    check_input(elements[0], elements[1], elements[2])
        else:
            raise ValueError("Incorrect input for community")

    if a is not None:
        if not (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)):
            raise ValueError("Provide a and b as DataFrames with row names or first column for individuals")
        if z is None:
            raise ValueError("z is empty; provide interaction matrix/data.frame")
