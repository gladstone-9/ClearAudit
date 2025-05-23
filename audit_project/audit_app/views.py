import csv
import io
import numpy as np
import pandas as pd
import os
import polars as pl
import opendp.prelude as dp
import plotly.graph_objects as go
import plotly.io as pio

from django.http import FileResponse
from django.views import generic
from django.http import JsonResponse
from django.http import HttpResponse
from django.views import View
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse

from .models import *

class PortalUploadView(View):

    def get(self, request):
        return render(request, 'portal_upload.html')

    def post(self, request):
        csv_file = request.FILES.get('csvFile')

        if csv_file:
            uploaded_file = UploadedFile(file=csv_file)
            uploaded_file.save()

            # Redirect to new view with the uploaded file's ID
            return redirect(reverse('explore_statistics', args=[uploaded_file.id]))

        return render(request, 'portal_upload.html', {"error": "No file uploaded"})
    
    
class ExploreStatisticsView(View):
    def get(self, request, file_id):
        uploaded_file = get_object_or_404(UploadedFile, id=file_id)

        # Read CSV content from saved file
        file = uploaded_file.file
        file.open(mode='r')  # Open in text mode
        text_data = file.read()  # This is already a string
        file.close()

        io_string = io.StringIO(text_data)
        reader = csv.reader(io_string)
        csv_data = list(reader)
        
        # Extract column names (assuming the first row contains the headers)
        columns = csv_data[0] if csv_data else []

        return render(request, "explore_statistics.html", {
            "uploaded_file": uploaded_file,
            "csv_data": csv_data,
            "file_id": file_id,
            "columns": columns
        })
        
# Return current epsilon and contributions values
def range_slider_view(request):
    try:
        epsilon = float(request.GET.get('epsilon', 1.0))
        # contributions = int(request.GET.get('contributions', 36))

        return JsonResponse({
            'epsilon': epsilon,
            # 'contributions': contributions
        })

    except ValueError as e:
        return JsonResponse({'error': 'Invalid input for epsilon or contributions.'}, status=400)


# DP Count
def dp_histogram_data(request):
    try:
        # Fields
        epsilon = float(request.GET.get('epsilon', 1.0))
        file_id = request.GET.get('file_id')
        label_column = request.GET.get('label_column')
        
        # True Stats
        df = get_data_frame(file_id)
        contributions = df.select(pl.col(label_column).n_unique()).collect().item()       # Unique records in col
        actual_count = df.select(pl.col(label_column).count()).collect().item()
        
        # DP Count Simulation
        scale = contributions / epsilon
        dp_counts = actual_count + np.random.laplace(loc=0, scale=scale, size=5000)

        return JsonResponse({
            'dp_counts': dp_counts.tolist(),
            'actual_count': actual_count
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
# DP sum
def dp_sum_data(request):
    try:
        # Fields
        epsilon = float(request.GET.get('epsilon', 1.0))
        target_column = request.GET.get('target_column')
        file_id = request.GET.get('file_id')
        label_column = request.GET.get('label_column')
        
        # True Stats
        df = get_data_frame(file_id)
        contributions = df.select(pl.col(label_column).n_unique()).collect().item()       # Unique records in col
        actual_sum = df.select(pl.col(target_column).sum()).collect().item()

        # DP Sum Simulation
        scale = contributions / epsilon
        dp_sums = actual_sum + np.random.laplace(loc=0, scale=scale, size=5000)

        return JsonResponse({
            'dp_sums': dp_sums.tolist(),
            'actual_sum': actual_sum
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def simulate_attack_view(request):
    # Fields
    epsilon = request.GET.get('epsilon')
    target_column = request.GET.get('target_column')
    file_id = request.GET.get('file_id')
    label_column = request.GET.get('label_column')
    
    # True Stats
    df = get_data_frame(file_id)
    contributions = df.select(pl.col(label_column).n_unique()).collect().item()       # Unique records in col

    return render(request, 'simulate_attacks.html', {
        'epsilon': epsilon,
        'contributions': contributions,
        'target_column': target_column,
        'file_id': file_id,
        'label_column': label_column,
    })

from django.http import JsonResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from diffprivlib.models import PCA as dpPCA
# from your_dp_library import dpPCA  # import your private PCA class

def pca_view(request):
    epsilon = float(request.GET.get('epsilon'))
    file_id = request.GET.get('file_id')
    label_column = request.GET.get('label_column')
    
    
    # Read and parse CSV
    df = get_non_lazy_data_frame(file_id)

    # Separate features and label
    y = df[label_column]
    # X = df.drop(columns=[label_column])
    X = df.drop(label_column, axis=1)

    # Convert features to numeric and clean data
    try:
        # Convert all features to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
                
        # Remove columns with all missing values
        X = X.dropna(axis=1, how='all')
        
        # Remove rows with any missing values
        X = X.dropna(axis=0)
        
        # Align target with cleaned features
        y = y.loc[X.index]
    except Exception as e:
        return JsonResponse({'error': f'Data conversion error: {str(e)}'}, status=400)

    # Check remaining data validity
    if X.empty or y.empty:
        return JsonResponse({'error': 'Not enough valid numeric data after cleaning.'}, status=400)

    # Standardize features
    try:
        X_scaled = StandardScaler().fit_transform(X)
        
    except Exception as e:
        return JsonResponse({'error': f'Feature scaling error: {str(e)}'}, status=400)

    # Perform PCA
    try:
        # Non-private PCA
        non_private_pca = PCA(n_components=2)
        X_np = non_private_pca.fit_transform(X_scaled)
        
        # Private PCA
        private_pca = dpPCA(epsilon=epsilon, n_components=2)
        X_p = private_pca.fit(X_scaled).transform(X_scaled)
    except Exception as e:
        return JsonResponse({'error': f'PCA calculation error: {str(e)}'}, status=400)

    # Convert numpy arrays to lists
    return JsonResponse({
        "non_private": X_np.tolist(),
        "private": X_p.tolist(),
        "labels": y.tolist()  # Fixed typo here
    })


def synthetic_data_view(request):
    # Fields
    epsilon = request.GET.get('epsilon')
    contributions = request.GET.get('contributions')
    target_column = request.GET.get('target_column')
    file_id = request.GET.get('file_id')
    label_column = request.GET.get('label_column')

    return render(request, 'generate_synthetic_data.html', {
        'epsilon': epsilon,
        'contributions': contributions,
        'target_column': target_column,
        'file_id': file_id,
        'label_column': label_column,
    })
    
def publish_data_release_view(request):
    # Fields
    epsilon = request.GET.get('epsilon')
    contributions = request.GET.get('contributions')
    target_column = request.GET.get('target_column')
    file_id = request.GET.get('file_id')
    label_column = request.GET.get('label_column')

    return render(request, 'publish_data_release.html', {
        'epsilon': epsilon,
        'contributions': contributions,
        'target_column': target_column,
        'file_id': file_id,
        'label_column': label_column,
    })

from django.http import FileResponse, Http404
def download_file(request):
    filename = request.GET.get('synthetic_filename')
    
    abs_base_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    synthetic_dir = os.path.join(abs_base_dir, "audit_project/media/uploads/synthetic_files")
    file_path = os.path.join(synthetic_dir, filename)

    if not os.path.exists(file_path):
        raise Http404("Synthetic file not found.")

    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)

import uuid
def create_synthetic_data(request):
    epsilon = float(request.GET.get('epsilon'))
    file_id = request.GET.get('file_id')
    target_column = request.GET.get('target_column')
    label_column = request.GET.get('label_column')
        
        
    df = get_non_lazy_data_frame(file_id)
    
    synthetic_df = generate_synthetic_data_new_environment(df, epsilon)
    
    # Temp Save Data
    unique_filename = f"synthetic_{uuid.uuid4().hex}.csv"
    abs_base_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    synthetic_dir = os.path.join(abs_base_dir, "audit_project/media/uploads/synthetic_files")
    os.makedirs(synthetic_dir, exist_ok=True)
    
    synthetic_path = os.path.join(synthetic_dir, unique_filename)
    synthetic_df.to_csv(synthetic_path, index=False)
    
    # Model Testing
    epsilons, baseline, accuracy_synth, accuracy_dp = draw_tradeoff(df, label_column, epsilon, "logreg")
    
    fig_json = create_plotly_figure(epsilons, baseline, accuracy_synth, accuracy_dp, title=f"Classification Accuaracy of {label_column}")
    
    # Store before this
    synthetic_df = synthetic_df.round(3)
    
    # Convert synthetic_df to CSV data
    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Move to start
    
    reader = csv.reader(csv_buffer)
    csv_data = list(reader)

    columns = csv_data[0] if csv_data else []

    return render(request, "generate_synthetic_data.html", {
        "csv_data": csv_data,
        "columns": columns,
        'epsilon': epsilon,
        'target_column': target_column,
        'file_id': file_id,
        'label_column': label_column,
        "synthetic_filename": unique_filename,
        "fig_json": fig_json,
    })
    
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from diffprivlib.models import LinearRegression as private_linearRegression
from diffprivlib.models import LogisticRegression as private_LogisticRegression
from sklearn.linear_model import LogisticRegression
from diffprivlib.models import RandomForestClassifier as private_RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# model can be one of ["linreg", "logreg", "randomforest"]
# the model decides if it is classification or regression
def draw_tradeoff(data, label_col, epsilon, model="linreg"):
    epsilons = np.logspace(0, 1, num=3, base=10)       # ie. 10^1 to 20^4, change num for more epsilons
    baseline = None
    accuracy_dp = []
    accuracy_synth = []

    dataX = data.drop(label_col, axis=1)
    dataY = data[label_col]
    #dataX = StandardScaler().fit_transform(dataX)
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)


    if model == "linreg":
        reg_model = lm.LinearRegression()
        dp_model = private_linearRegression
    elif model == "logreg":
        reg_model = LogisticRegression()
        dp_model = private_LogisticRegression
    elif model == "randomforest":
        reg_model = RandomForestClassifier()
        dp_model = private_RandomForestClassifier
    else:
        raise Exception("Invalid model")

    reg_model.fit(X_train, y_train)
    baseline = reg_model.score(X_test, y_test)
    print(f"Baseline: {baseline}")

    for epsilon in epsilons:
        dp_model_temp = dp_model(epsilon=epsilon)
        dp_model_temp.fit(X_train, y_train)

        accuracy_dp.append(dp_model_temp.score(X_test, y_test))

        try:
            # synth = Synthesizer.create('mwem', epsilon=epsilon, split_factor=5) # we need the split factor so we don't run out of memory!!
            # synth = Synthesizer.create('mst', epsilon=epsilon)
            # synth_train = synth.fit_sample(X_train, preprocessor_eps=2.0) #make this a slider
            synth_train = generate_synthetic_data_new_environment(X_train, epsilon)
            print("train", X_train.info())
            print("synth", synth_train.info())
            reg_model.fit(synth_train, y_train)

            accuracy_synth.append(reg_model.score(X_test, y_test))
        except Exception as error:
            print("failed because", error)
            accuracy_synth.append(0.0)
        #break
        
    return epsilons, baseline, accuracy_synth, accuracy_dp

    # plt.semilogx(epsilons, accuracy_dp, label="Differentially private linear regression", zorder=10)
    # plt.semilogx(epsilons, accuracy_synth, label="Synthetic data", zorder=5)
    # plt.semilogx(epsilons, baseline * np.ones_like(epsilons), dashes=[2,2], label="Non-private baseline", zorder=5)
    # plt.xlabel("epsilon")
    # plt.ylabel("loss")
    # plt.ylim(-3*abs(baseline), 3*abs(baseline))
    # plt.legend()    

def create_plotly_figure(epsilons, baseline, accuracy_synth, accuracy_dp, title=""):
    # Create the Plotly figure
    fig = go.Figure()

    # DP Linear Regression Accuracy
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=accuracy_dp,
        mode='lines+markers',
        name='Differentially private logistic regression',
        line=dict(width=3),
    ))

    # Synthetic Data Accuracy
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=accuracy_synth,
        mode='lines+markers',
        name='Synthetic data',
        line=dict(width=2, dash='solid'),
    ))

    # Non-private Baseline (horizontal dashed line)
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=[baseline] * len(epsilons),
        mode='lines',
        name='Non-private baseline',
        line=dict(dash='dash', width=2),
    ))

    # Layout Settings
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='Epsilon',
            type='log',
            tickvals=epsilons,  # <-- force ticks only at your points
            ticktext=[str(int(e)) for e in epsilons],  # <-- make labels exactly 10, 100, 1000, 10000
        ),
        yaxis=dict(title='Loss', range=[-3*abs(baseline), 3*abs(baseline)]),
        legend=dict(title="Models"),
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Serialize figure to JSON
    fig_json = pio.to_json(fig)
    return fig_json

def publish_stats(request):
    # Fields
    epsilon = float(request.GET.get('epsilon'))
    target_column = request.GET.get('target_column')
    file_id = int(request.GET.get('file_id'))
    label_column = request.GET.get('label_column')
        
    NUM_QUERIES = 2 # Number of planned general statistics (may change)

    # True Stats
    df = get_data_frame(file_id)
    contributions = df.select(pl.col(label_column).n_unique()).collect().item()     # Change this to label column
    true_count = df.select(pl.col(target_column).count()).collect().item()
    true_sum = df.select(pl.col(target_column).sum()).collect().item()
    true_mean = df.select(pl.col(target_column).mean()).collect().item()
    
    print(type(df))
    
    dp.enable_features("contrib")
    context = dp.Context.compositor(
        data=df,
        privacy_unit=dp.unit_of(contributions=contributions),     # len Unique Rows in label column  
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=NUM_QUERIES,
        margins=[
            dp.polars.Margin(
                max_partition_length=true_count # the biggest partition
            ),
        ],
    )
    
    # DP Count
    query_count = context.query().select(dp.len())
    query_count_val = query_count.release().collect().item()
    
    # DP Sum (of target column)       
    max_val = df.select(pl.col(target_column).max()).collect().item()
    min_val = df.select(pl.col(target_column).min()).collect().item()
    
    query_sum = (
        context.query()
        # Compute the DP sum for the target column
        .select(pl.col(target_column).cast(int).fill_null(true_mean).dp.sum(bounds=(min_val, max_val)))
    )
    query_sum_val = query_sum.release().collect().item()
    
    # Mean
    query_mean_val = query_sum_val / query_count_val
    
    return render(request, 'publish_data_release.html', {
        'epsilon': epsilon,
        'contributions': contributions,
        'target_column': target_column,
        'label_column': label_column,
        'file_id': file_id,
        'query_count': f"{query_count_val:,}",
        'query_sum': f"{query_sum_val:,.3f}",
        'query_mean': f"{query_mean_val:,.3f}",
        'true_count': f"{true_count:,}",
        'true_mean': f"{true_mean:,.3f}",
        'true_sum': f"{true_sum:,.3f}",
    })    

def get_data_frame(file_id):
    uploaded_file = get_object_or_404(UploadedFile, id=file_id)
    filepath = uploaded_file.file.path 
    data = pl.scan_csv(filepath, ignore_errors=True)
    return data

def get_non_lazy_data_frame(file_id):
    uploaded_file = get_object_or_404(UploadedFile, id=file_id)
    data = pd.read_csv(uploaded_file.file)
    return data

import subprocess
import pickle
import tempfile
from pathlib import Path
# Generate Synthetic Data with a older version of opendp (in a new environment)
def generate_synthetic_data_new_environment(df, needed_epsilon):
    # subprocess.run(["source", "synthetic_data_venv/bin/activate", "&&", "python", "synthetic_data.py"])
    
    # Save the input DataFrame to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_input:
        input_path = tmp_input.name
        with open(input_path, "wb") as f:
            pickle.dump(df, f)

    # Prepare output temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_output:
        output_path = tmp_output.name
    
    # Set paths
    abs_base_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    other_venv_python = os.path.join(abs_base_dir, "synthetic_data_venv/bin/python")
    other_script = os.path.join(abs_base_dir, "audit_project/audit_app/synthetic_data.py")
    
    # Run the subprocess
    subprocess.run([
        other_venv_python,
        other_script,  # Path to your helper script
        input_path,
        str(needed_epsilon),
        output_path
    ], check=True)
    
    # Load the resulting synthetic DataFrame
    with open(output_path, "rb") as f:
        synthetic_df = pickle.load(f)
        
    return synthetic_df
