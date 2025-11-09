import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from skimage import measure

import warnings
warnings.filterwarnings("ignore") # Added this at the end to show a clean output with no warnings but not necessary

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Function to extract features (area and perimeter)
def extract_features(images):
    features = []
    for img in images:
        # threshold the image
        thres_img = 1*(img > 0)

        # Get region properties
        props = measure.regionprops(measure.label(thres_img))

        if len(props) > 0:
            # Use the first (largest) region
            area = props[0].area
            perimeter = props[0].perimeter
            features.append([area, perimeter])
        else:
            # If no region found, use zeros
            features.append([0, 0])

    return np.array(features)

def plot_misclassified(x_train_new, y_train_new, y_pred_train, first_num, sec_num, test=False):
    """
    Plot average misclassified images by class and show individual examples.
    
    Arguments:
    x_train_new -- training images (original 28x28 images)
    y_train_new -- true labels
    y_pred_train -- predicted labels
    first_num -- first class number
    sec_num -- second class number
    """
    # Find misclassified samples
    misclassified_mask = (y_pred_train != y_train_new)
    misclassified_images = x_train_new[misclassified_mask]
    misclassified_true_labels = y_train_new[misclassified_mask]
    misclassified_pred_labels = y_pred_train[misclassified_mask]
    
    if test:
        print(f"Number of misclassified testing samples: {np.sum(misclassified_mask)}")
    else:
      print(f"Number of misclassified training samples: {np.sum(misclassified_mask)}")
    
    # Separate misclassified samples by true class
    if np.sum(misclassified_mask) > 0:
        misclassified_class0 = misclassified_images[misclassified_true_labels == first_num]
        misclassified_class1 = misclassified_images[misclassified_true_labels == sec_num]
        
        print(f"Misclassified {first_num}'s: {len(misclassified_class0)}")
        print(f"Misclassified {sec_num}'s: {len(misclassified_class1)}")
        
        # Plot average of misclassified images for each class
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Average of misclassified class 0
        if len(misclassified_class0) > 0:
            avg_misclassified_class0 = np.mean(misclassified_class0, axis=0)
            im0 = axes[0].imshow(avg_misclassified_class0, cmap='gray')
            axes[0].set_title(f'Average of Misclassified {first_num}\'s\n(n={len(misclassified_class0)})')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0])
        else:
            axes[0].text(0.5, 0.5, f'No misclassified {first_num}\'s', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].axis('off')
        
        # Average of misclassified class 1
        if len(misclassified_class1) > 0:
            avg_misclassified_class1 = np.mean(misclassified_class1, axis=0)
            im1 = axes[1].imshow(avg_misclassified_class1, cmap='gray')
            axes[1].set_title(f'Average of Misclassified {sec_num}\'s\n(n={len(misclassified_class1)})')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, f'No misclassified {sec_num}\'s', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')
        
        if test:
            plt.suptitle(f'Average Misclassified Testing Images ({first_num} vs {sec_num})')
        else:
            plt.suptitle(f'Average Misclassified Training Images ({first_num} vs {sec_num})')

        plt.tight_layout()
        plt.show()
        
        # Show individual examples of misclassified images
        num_examples = min(5, np.sum(misclassified_mask))
        if num_examples > 0:
            fig, axes = plt.subplots(1, num_examples, figsize=(num_examples*2, 2))
            if num_examples == 1:
                axes = [axes]
            
            for i in range(num_examples):
                axes[i].imshow(misclassified_images[i], cmap='gray')
                axes[i].set_title(f'True: {misclassified_true_labels[i]}\nPred: {misclassified_pred_labels[i]}')
                axes[i].axis('off')
            
            if test:
                plt.suptitle('Examples of Misclassified Testing Images')
            else:
                plt.suptitle('Examples of Misclassified Training Images')
            
            plt.tight_layout()
            plt.show()
    else:
        print("No misclassified samples to plot!")
    
    return

def FisherDis(first_num, sec_num):
    #Data
    data1 = x_train[y_train==first_num]
    data2 = x_train[y_train==sec_num]

    numberSamplesTest1 = x_test[y_test==first_num]
    numberSamplesTest2 = x_test[y_test==sec_num]

    print(f"Number of test images for class {first_num} = {numberSamplesTest1.shape}")
    print(f"Number of test images for class {sec_num} = {numberSamplesTest2.shape}")

    #Feature extreaction: ONLY CONTAINS (Area, Perimeter)
    features1 = extract_features(data1)
    features2 = extract_features(data2)

    print(f"Number of {first_num}'s: {features1.shape[0]}")
    print(f"Number of {sec_num}'s: {features2.shape[0]}")
    print(f"Feature shape for {first_num}: {features1.shape}")
    print(f"Feature shape for {sec_num}: {features2.shape}")

    #Mean
    m1 = np.mean(features1, axis=0)
    m2 = np.mean(features2, axis=0)

    print(f"Mean features for {first_num}: Area={m1[0]:.2f}, Perimeter={m1[1]:.2f}")
    print(f"Mean features for {sec_num}: Area={m2[0]:.2f}, Perimeter={m2[1]:.2f}")

    #Scatter
    scatter1 = np.cov(features1, rowvar=False)
    scatter2 = np.cov(features2, rowvar=False)

    #Within-class scatter
    sw = scatter1 + scatter2

    # Discriminant vector
    w = np.linalg.pinv(sw) @ (m2 - m1)
    print(f"Discriminant vector w: {w}")

    #Project w onto training data
    x_train_new = x_train[(y_train==first_num) | (y_train==sec_num)]
    x_train_new_features = extract_features(x_train_new)
    x_train_new_projections = x_train_new_features @ w

    # Project w onto test data
    x_test_new = x_test[(y_test==first_num) | (y_test==sec_num)]
    x_test_new_features = extract_features(x_test_new)
    x_test_new_projections = x_test_new_features @ w

    # Threshold
    y_train_new = y_train[(y_train == first_num) | (y_train == sec_num)]
    new_m1 = np.mean(x_train_new_projections[y_train_new == first_num])
    new_m2 = np.mean(x_train_new_projections[y_train_new == sec_num])
    thres = (new_m1 + new_m2) / 2
    print(f"Threshold: {thres:.4f}")

    # Predict Training
    y_pred_train = (x_train_new_projections > thres).astype(int)
    # Map predictions: 0 -> first_num, 1 -> sec_num
    y_pred_train = np.where(y_pred_train == 0, first_num, sec_num)
    accuracy_train = np.mean(y_pred_train == y_train_new)
    print(f"Train Data Accuracy: {accuracy_train*100:.2f}%")

    #---------------------------------------------------------------
    #Plotting misclassifications
    plot_misclassified(x_train_new, y_train_new, y_pred_train, first_num, sec_num, False)

    # Predict Test
    y_test_new = y_test[(y_test == first_num) | (y_test == sec_num)]
    y_pred_test = (x_test_new_projections > thres).astype(int)
    # Map predictions: 0 -> first_num, 1 -> sec_num
    y_pred_test = np.where(y_pred_test == 0, first_num, sec_num)
    accuracy_test = np.mean(y_pred_test == y_test_new)
    print(f"Test Data Accuracy: {accuracy_test*100:.2f}%")
    print()

    #---------------------------------------------------------------
    #Plotting misclassifications
    plot_misclassified(x_test_new, y_test_new, y_pred_test, first_num, sec_num, True)

    return
    


#-----------------------------------------------------------------
#for numbers 0 & 1
FisherDis(0,1)
#-----------------------------------------------------------------
#for numbers 5 & 6
FisherDis(5,6)
