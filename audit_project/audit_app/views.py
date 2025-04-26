import csv
import io
import numpy as np
import pandas as pd

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
        contributions = int(request.GET.get('contributions', 36))

        return JsonResponse({
            'epsilon': epsilon,
            'contributions': contributions
        })

    except ValueError as e:
        return JsonResponse({'error': 'Invalid input for epsilon or contributions.'}, status=400)


# DP Count
def dp_histogram_data(request):
    try:
        epsilon = float(request.GET.get('epsilon', 1.0))
        contributions = int(request.GET.get('contributions', 36))

        if epsilon <= 0:
            return JsonResponse({'error': 'Epsilon must be greater than 0.'}, status=400)

        # Get actual count (# of records)
        file_id = request.GET.get('file_id')
        uploaded_file = get_object_or_404(UploadedFile, id=file_id)

        # Read CSV content from saved file
        file = uploaded_file.file
        file.open(mode='r')  # Open in text mode
        text_data = file.read()  # This is already a string
        file.close()

        io_string = io.StringIO(text_data)
        reader = csv.reader(io_string)
        csv_data = list(reader)

        # Assuming the CSV has headers and we count the rows, excluding the header
        actual_count = len(csv_data) - 1  # Subtract 1 for the header row
                
        scale = contributions / epsilon
        dp_counts = actual_count + np.random.laplace(loc=0, scale=scale, size=5000)

        return JsonResponse({
            'dp_counts': dp_counts.tolist(),
            'actual_count': actual_count
        })

    except ValueError:
        return JsonResponse({'error': 'Invalid numeric input.'}, status=400)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
# DP sum
def dp_sum_data(request):
    try:
        epsilon = float(request.GET.get('epsilon', 1.0))
        contributions = int(request.GET.get('contributions', 36))
        max_bound = float(request.GET.get('max_bound', 100))
        target_column = request.GET.get('target_column')
        file_id = request.GET.get('file_id')

        if not target_column:
            return JsonResponse({'error': 'Target column is required.'}, status=400)
        if epsilon <= 0:
            return JsonResponse({'error': 'Epsilon must be greater than 0.'}, status=400)

        uploaded_file = get_object_or_404(UploadedFile, id=file_id)

        # Read file
        file = uploaded_file.file
        file.open(mode='r')
        text_data = file.read()
        file.close()

        io_string = io.StringIO(text_data)
        reader = csv.DictReader(io_string)
        values_clipped = []
        values = []

        for row in reader:
            try:
                val = float(row[target_column])
                clipped = min(val, max_bound)
                values.append(val)
                values_clipped.append(clipped)
            except (ValueError, KeyError):
                continue  # Skip rows with missing or non-numeric data

        actual_sum = sum(values)
        scale = contributions / epsilon
        dp_sums = actual_sum + np.random.laplace(loc=0, scale=scale, size=5000)
        dp_sums = dp_sums[dp_sums <= max_bound]    # Enforce max bound

        return JsonResponse({
            'dp_sums': dp_sums.tolist(),
            'actual_sum': actual_sum
        })

    except ValueError:
        return JsonResponse({'error': 'Invalid numeric input.'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def simulate_attack_view(request):
    # Get the parameters from the URL
    epsilon = request.GET.get('epsilon')
    contributions = request.GET.get('contributions')
    max_bound = request.GET.get('max_bound')
    target_column = request.GET.get('target_column')
    file_id = request.GET.get('file_id')

    # Process the parameters (add any processing logic you need here)

    # Return the response, either rendering a template or returning data (e.g., JSON)
    return render(request, 'simulate_attacks.html', {
        'epsilon': epsilon,
        'contributions': contributions,
        'max_bound': max_bound,
        'target_column': target_column,
        'file_id': file_id,
    })

from django.http import JsonResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from diffprivlib.models import PCA as dpPCA
# from your_dp_library import dpPCA  # import your private PCA class

def pca_view(request):
    epsilon = float(request.GET.get('epsilon', 1.0))
    file_id = request.GET.get('file_id')
    target_column = request.GET.get('column')
    
    # Validate inputs
    if not target_column:
        return JsonResponse({'error': 'Target column is required.'}, status=400)
    if epsilon <= 0:
        return JsonResponse({'error': 'Epsilon must be greater than 0.'}, status=400)

    # Get uploaded file
    uploaded_file = get_object_or_404(UploadedFile, id=file_id)
    
    # Read and parse CSV
    try:
        df = pd.read_csv(uploaded_file.file)
    except Exception as e:
        return JsonResponse({'error': f'Error reading CSV file: {str(e)}'}, status=400)

    # Check target column exists
    if target_column not in df.columns:
        return JsonResponse({'error': f'Column "{target_column}" not found in file.'}, status=400)

    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])

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
    # Get the parameters from the URL
    epsilon = request.GET.get('epsilon')
    contributions = request.GET.get('contributions')
    max_bound = request.GET.get('max_bound')
    target_column = request.GET.get('target_column')
    file_id = request.GET.get('file_id')

    # Process the parameters (add any processing logic you need here)

    # Return the response, either rendering a template or returning data (e.g., JSON)
    return render(request, 'generate_synthetic_data.html', {
        'epsilon': epsilon,
        'contributions': contributions,
        'max_bound': max_bound,
        'target_column': target_column,
        'file_id': file_id,
    })
    
def publish_data_release_view(request):
        # Get the parameters from the URL
    epsilon = request.GET.get('epsilon')
    contributions = request.GET.get('contributions')
    max_bound = request.GET.get('max_bound')
    target_column = request.GET.get('target_column')
    file_id = request.GET.get('file_id')

    # Process the parameters (add any processing logic you need here)

    # Return the response, either rendering a template or returning data (e.g., JSON)
    return render(request, 'publish_data_release.html', {
        'epsilon': epsilon,
        'contributions': contributions,
        'max_bound': max_bound,
        'target_column': target_column,
        'file_id': file_id,
    })