// MARK: - Imports: Frameworks for UI, camera, vision, image processing, and reactive binding
import SwiftUI            // SwiftUI for declarative UI
import UIKit               // UIKit for view controllers and UI elements
import AVFoundation        // AVFoundation for camera capture
import Vision              // Vision for image analysis (contour detection)
import CoreImage           // CoreImage for image handling (if needed)
import Combine             // Combine for reactive bindings

// MARK: - ContentView: Wraps a UIKit controller in SwiftUI
struct ContentView: UIViewControllerRepresentable {
    // Bindings to hold user-entered parameters
    @Binding var pipetDiameter: String       // Pipette diameter in millimeters
    @Binding var density: String             // Liquid density in kg/m³

    // MARK: - Coordinator: Handles camera session, photo capture, and analysis
    class Coordinator: NSObject, AVCapturePhotoCaptureDelegate, ObservableObject {
        let session = AVCaptureSession()        // Manages camera input/output
        let output = AVCapturePhotoOutput()     // Photo output for still captures
        var parent: ContentView                 // Reference back to ContentView
        @Published var surfaceTension: String = "Tap to measure"  // Published result
        var photoCompletion: ((UIImage) -> Void)?
        var cancellables = Set<AnyCancellable>()

        // Coordinator initializer: store parent, then configure camera
        init(parent: ContentView) {
            self.parent = parent
            super.init()
            setupCamera()       // Configure session inputs/outputs
            checkPermissions()  // Request camera access if needed
        }

        // Configure camera device, focus, exposure, and attach to session
        func setupCamera() {
            session.sessionPreset = .high  // High-quality capture

            // Obtain the back wide-angle camera
            guard let device = AVCaptureDevice.default(
                    .builtInWideAngleCamera,
                    for: .video,
                    position: .back),
                  let input = try? AVCaptureDeviceInput(device: device)
            else {
                print("Failed to get camera input")
                return
            }

            // Lock configuration to set focus/exposure
            do {
                try device.lockForConfiguration()

                // Continuous autofocus centered in view
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
                    device.focusMode = .continuousAutoFocus
                }

                // Continuous autoexposure centered in view
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5)
                    device.exposureMode = .continuousAutoExposure
                }

                // Optionally lock lens position for close-up
                if device.isLockingFocusWithCustomLensPositionSupported {
                    let desiredLensPosition: Float = 0.1
                    device.setFocusModeLocked(lensPosition: desiredLensPosition, completionHandler: nil)
                }

                device.unlockForConfiguration()
            } catch {
                print("Error configuring device: \(error)")
            }

            // Add input and output to the session
            if session.canAddInput(input) {
                session.addInput(input)
            }
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
        }

        // Check and request camera permissions, then start session
        func checkPermissions() {
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            switch status {
            case .authorized:
                // Already authorized: start running
                DispatchQueue.global(qos: .userInitiated).async {
                    self.session.startRunning()
                }
            case .notDetermined:
                // Request access
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        DispatchQueue.global(qos: .userInitiated).async {
                            self.session.startRunning()
                        }
                    } else {
                        DispatchQueue.main.async {
                            self.surfaceTension = "Camera access denied"
                        }
                    }
                }
            default:
                // Denied or restricted
                self.surfaceTension = "Camera access denied"
            }
        }

        // Button action: capture a photo and handle analysis
        @objc func capturePhoto() {
            print("capturePhoto triggered")
            let settings = AVCapturePhotoSettings()

            // Set completion closure to process the captured image
            photoCompletion = { image in
                DispatchQueue.main.async {
                    let result = self.calculateSurfaceTension(
                        from: image,
                        pipetDiameter: self.parent.pipetDiameter,
                        density: self.parent.density)
                    print("Analysis result: \(String(describing: result))")
                    self.surfaceTension = result ?? "Measurement failed"
                }
            }

            // Trigger capture
            output.capturePhoto(with: settings, delegate: self)
        }

        // Delegate callback when photo processing completes
        func photoOutput(_ output: AVCapturePhotoOutput,
                         didFinishProcessingPhoto photo: AVCapturePhoto,
                         error: Error?) {
            if let error = error {
                print("Error capturing photo: \(error.localizedDescription)")
                return
            }
            // Convert photo data to UIImage
            guard let data = photo.fileDataRepresentation(),
                  let image = UIImage(data: data) else {
                print("Failed to convert image data")
                return
            }
            // Invoke completion closure
            photoCompletion?(image)
        }

        /// Analyze the image, detect droplet contour, compute scale and surface tension
        func calculateSurfaceTension(
            from image: UIImage,
            pipetDiameter: String,
            density: String
        ) -> String? {
            // Get CGImage for Vision handler
            guard let cgImage = image.cgImage else {
                print("No CGImage found in image")
                return nil
            }

            // Set up Vision request for contour detection
            let request = VNDetectContoursRequest()
            request.contrastAdjustment = 1.0
            request.detectDarkOnLight = true
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            do {
                // Perform contour detection
                try handler.perform([request])
                guard let observation = request.results?.first as? VNContoursObservation else {
                    print("No contour observation found")
                    return nil
                }
                // Choose largest contour (assumed droplet)
                let contours = observation.topLevelContours
                guard let droplet = contours.max(by: { $0.pointCount < $1.pointCount }) else {
                    print("No droplet contour found")
                    return nil
                }
                let points = droplet.normalizedPoints

                // Compute bounding box (min/max of normalized points)
                let xs = points.map { CGFloat($0.x) }
                let ys = points.map { CGFloat($0.y) }
                guard let minX = xs.min(), let maxX = xs.max(),
                      let minY = ys.min(), let maxY = ys.max() else {
                    print("Failed to determine bounding box")
                    return nil
                }
                let widthNorm = maxX - minX
                let heightNorm = maxY - minY

                // Determine scale factor: mm per pixel using pipetDiameter input
                var scaleFactor: CGFloat = 0.01  // default fallback
                if let d = Double(pipetDiameter), d > 0 {
                    let pixelWidth = widthNorm * CGFloat(cgImage.width)
                    scaleFactor = CGFloat(d) / pixelWidth
                }

                // Compute droplet size in mm
                let dropletWidthMM = widthNorm * CGFloat(cgImage.width) * scaleFactor
                let dropletHeightMM = heightNorm * CGFloat(cgImage.height) * scaleFactor
                print("Measured droplet width: \(dropletWidthMM) mm, height: \(dropletHeightMM) mm")

                // Parse density input or default to water
                let rho: Double = (Double(density) ?? 1000.0)
                let g = 9.81  // m/s²

                // Convert mm to meters
                let de = Double(dropletWidthMM) / 1000.0
                let h  = Double(dropletHeightMM) / 1000.0

                // Surface tension formula: γ = ρ·g·L·h
                let tension = rho * g * de * h

                // Return formatted in mN/m
                return String(format: "Surface Tension: %.1f mN/m", tension * 1000)
            } catch {
                print("Error performing contour request: \(error.localizedDescription)")
                return nil
            }
        }
    } // End Coordinator

    // Create Coordinator instance
    func makeCoordinator() -> Coordinator {
        return Coordinator(parent: self)
    }

    // Create the UIKit controller with preview, button, and label
    func makeUIViewController(
        context: UIViewControllerRepresentableContext<ContentView>
    ) -> UIViewController {
        let vc = UIViewController()

        // Camera preview layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: context.coordinator.session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = vc.view.bounds
        vc.view.layer.addSublayer(previewLayer)

        // Capture button setup
        let button = UIButton(type: .system)
        button.setTitle("Start Analysis", for: .normal)
        button.backgroundColor = .systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 8
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addTarget(
            context.coordinator,
            action: #selector(context.coordinator.capturePhoto),
            for: .touchUpInside
        )
        vc.view.addSubview(button)
        NSLayoutConstraint.activate([
            button.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            button.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            button.bottomAnchor.constraint(equalTo: vc.view.bottomAnchor, constant: -120),
            button.heightAnchor.constraint(equalToConstant: 40)
        ])

        // Result label to display surface tension
        let label = UILabel()
        label.textAlignment = .center
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.layer.cornerRadius = 8
        label.clipsToBounds = true
        label.translatesAutoresizingMaskIntoConstraints = false
        vc.view.addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            label.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            label.bottomAnchor.constraint(equalTo: vc.view.bottomAnchor, constant: -60),
            label.heightAnchor.constraint(equalToConstant: 40)
        ])

        // Bind published surfaceTension to label text
        context.coordinator.$surfaceTension
            .receive(on: DispatchQueue.main)
            .sink { text in label.text = text }
            .store(in: &context.coordinator.cancellables)

        return vc
    }

    // No per-update logic needed
    func updateUIViewController(_ uiViewController: UIViewController, context: UIViewControllerRepresentableContext<ContentView>) {}
}

// MARK: - MainView: Root SwiftUI view with user inputs
struct MainView: View {
    @State private var pipetDiameter: String = ""  // Diameter field
    @State private var density: String = ""        // Density field

    var body: some View {
        VStack {
            // Text field for pipette diameter
            TextField("Enter pipet diameter (mm)", text: $pipetDiameter)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
                .keyboardType(.decimalPad)
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Done") {
                            UIApplication.shared.sendAction(
                                #selector(UIResponder.resignFirstResponder),
                                to: nil, from: nil, for: nil)
                        }
                    }
                }
            // Text field for liquid density
            TextField("Enter density (kg/m³)", text: $density)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding([.horizontal, .bottom])
                .keyboardType(.decimalPad)
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Done") {
                            UIApplication.shared.sendAction(
                                #selector(UIResponder.resignFirstResponder),
                                to: nil, from: nil, for: nil)
                        }
                    }
                }
            // Embed the camera/analysis view
            ContentView(pipetDiameter: $pipetDiameter, density: $density)
                .edgesIgnoringSafeArea(.all)
        }
    }
}

// Preview provider for SwiftUI canvas
struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
}
