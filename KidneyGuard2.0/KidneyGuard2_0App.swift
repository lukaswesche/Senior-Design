//
//  KidneyGuard2_0App.swift
//  KidneyGuard2.0
//
//  Created by Zach Tinsley on 4/25/25.
//

import SwiftUI

@main
struct KidneyGuard2_0App: App {
    // Provide @State storage for the two bindings
    @State private var pipetDiameter: String = ""
    @State private var density: String = ""

    var body: some Scene {
        WindowGroup {
            // Pass the bindings into ContentView
            ContentView(pipetDiameter: $pipetDiameter,
                        density: $density)
        }
    }
}
