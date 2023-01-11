/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller for selecting images and applying Vision + Core ML processing.
*/

import UIKit
import CoreML
import Vision
import ImageIO

class ImageClassificationViewController: UIViewController {
    // MARK: - IBOutlets
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var cameraButton: UIBarButtonItem!
    @IBOutlet weak var classificationLabel: UILabel!
    
    let aom_model: VNCoreMLModel;
    let csom_model: VNCoreMLModel;
    
    required init(coder: NSCoder) {
        do{
            let model_config = MLModelConfiguration();
            model_config.computeUnits = .all;
            self.aom_model = try VNCoreMLModel(for: AOM(configuration: model_config).model);
            self.csom_model = try VNCoreMLModel(for: CSOM(configuration: model_config).model);
            super.init(coder: coder)!
        }catch{
            fatalError("Failed to load ML Model. Error: \(error)")
        }
    }
    
    /// - Tag: MLModelSetup
    lazy var classificationRequests: [VNCoreMLRequest] = {
        var requests: [VNCoreMLRequest] = [];
        var request: VNCoreMLRequest;
        /*
         Use the Swift class `MobileNet` Core ML generates from the model.
         To use a different Core ML classifier model, add it to the project
         and replace `MobileNet` with that model's generated Swift class.
         */
        request = VNCoreMLRequest(model: aom_model, completionHandler: { [weak self] request, error in
            self?.processClassifications(for: request, error: error, tag: "Acute Otitis Media")
        })
        request.imageCropAndScaleOption = .centerCrop
        requests.append(request);
        
        request = VNCoreMLRequest(model: csom_model, completionHandler: { [weak self] request, error in
            self?.processClassifications(for: request, error: error, tag: "Chronic Suppurative Otitis Media")
        })
        request.imageCropAndScaleOption = .centerCrop
        requests.append(request);
        
        return requests
    }()
    
    /// - Tag: PerformRequests
    func updateClassifications(image: UIImage) {
        classificationLabel.text = "Classifying..."
        
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                for classificationRequest in self.classificationRequests{
                    try handler.perform([classificationRequest])
                }
            } catch {
                /*
                 This handler catches general image processing errors. The `classificationRequest`'s
                 completion handler `processClassifications(_:error:)` catches errors specific
                 to processing that request.
                 */
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    
    /// Updates the UI with the results of the classification.
    /// - Tag: ProcessClassifications
    func processClassifications(for request: VNRequest, error: Error?, tag: String) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.classificationLabel.text = "An error has occured."
                return
            }
            // The `results` will always be `VNClassificationObservation`s, as specified by the Core ML model in this project.
            let classification = results[0] as! VNCoreMLFeatureValueObservation
            let featureValue = classification.featureValue
            let value = featureValue.multiArrayValue![0]
            if self.classificationLabel.text == "Classifying..." {
                self.classificationLabel.text = "\(tag): \(value.decimalValue*100.0)%";
            }else{
                self.classificationLabel.text! += "\n\(tag): \(value.decimalValue*100.0)%";
            }
        }
    }
    
    // MARK: - Allow User to take Photos
    
    @IBAction func takePicture() {
        // Show options for the source picker only if the camera is available.
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            presentPhotoPicker(sourceType: .photoLibrary)
            return
        }
        
        let photoSourcePicker = UIAlertController(title: "Take or Choose an Image", message: "Please use the included otoscope to take a photo with your camera, or choose one you have already taken from your photo library.", preferredStyle: UIAlertController.Style.alert)
        let takePhoto = UIAlertAction(title: "Take Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .camera)
        }
        let choosePhoto = UIAlertAction(title: "Choose Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .photoLibrary)
        }
        
        photoSourcePicker.addAction(takePhoto)
        photoSourcePicker.addAction(choosePhoto)
        photoSourcePicker.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        
        present(photoSourcePicker, animated: true)
    }
    
    func presentPhotoPicker(sourceType: UIImagePickerController.SourceType) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = sourceType
        present(picker, animated: true)
    }
}

extension ImageClassificationViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // MARK: - Handling Image Picker Selection

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {

        picker.dismiss(animated: true)
        
        // We always expect `imagePickerController(:didFinishPickingMediaWithInfo:)` to supply the original image.
        let image = info[UIImagePickerController.InfoKey.originalImage] as! UIImage
        imageView.image = image
        updateClassifications(image: image)
    }
}
