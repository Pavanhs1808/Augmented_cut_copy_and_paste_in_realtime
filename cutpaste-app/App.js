import React, { useState, useEffect } from 'react';
import { Button, Image, View, StyleSheet, ActivityIndicator, Alert, Text, SafeAreaView, ScrollView, Animated, TouchableOpacity, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { Camera } from 'expo-camera';

export default function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [selectedObject, setSelectedObject] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [cameraPermission, setCameraPermission] = useState(null);
  const [mediaPermission, setMediaPermission] = useState(null);
  const [showAR, setShowAR] = useState(false);
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 700,
      useNativeDriver: true,
    }).start();
  }, []);

  useEffect(() => {
    (async () => {
      const cameraStatus = await ImagePicker.getCameraPermissionsAsync();
      const mediaStatus = await ImagePicker.getMediaLibraryPermissionsAsync();
      console.log('Camera:', cameraStatus.status, 'Media:', mediaStatus.status);
      setCameraPermission(cameraStatus.status);
      setMediaPermission(mediaStatus.status);
    })();
  }, []);

  const requestPermissions = async () => {
    try {
      const cameraStatus = await ImagePicker.requestCameraPermissionsAsync();
      const mediaStatus = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      console.log('Requested - Camera:', cameraStatus.status, 'Media:', mediaStatus.status);
      
      setCameraPermission(cameraStatus.status);
      setMediaPermission(mediaStatus.status);
      
      if (cameraStatus.status !== 'granted' || mediaStatus.status !== 'granted') {
        Alert.alert(
          'Permission Required', 
          'Please grant camera and media library permissions in your device settings to use all features.',
          [
            { text: 'OK', style: 'default' }
          ]
        );
      } else {
        Alert.alert('Success', 'All permissions granted!');
      }
    } catch (error) {
      console.error('Permission request error:', error);
      Alert.alert('Error', 'Failed to request permissions');
    }
  };

  const pickImage = async () => {
    if (mediaPermission !== 'granted') {
      Alert.alert(
        'Permission Required', 
        'Media Library permission is required to pick images.',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Grant Permission', onPress: requestPermissions }
        ]
      );
      return;
    }
    
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: 'images',
        allowsEditing: false, // Don't edit to preserve original for detection
        quality: 0.9, // Higher quality for better detection
      });
      
      if (!result.canceled && result.assets && result.assets.length > 0) {
        setOriginalImage(result.assets[0].uri);
        setProcessedImage(null);
        setDetectedObjects([]);
        setSelectedObject(null);
        console.log('Image selected from gallery');
        
        // Automatically detect objects after image selection
        detectObjects(result.assets[0].uri);
      }
    } catch (err) {
      console.error('Gallery error:', err);
      Alert.alert('Error', 'Could not open gallery. Please try again.');
    }
  };

  const takePhoto = async () => {
    if (cameraPermission !== 'granted') {
      Alert.alert(
        'Permission Required', 
        'Camera permission is required to take photos.',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Grant Permission', onPress: requestPermissions }
        ]
      );
      return;
    }
    
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: 'images',
        allowsEditing: false, // Don't edit to preserve original for detection
        quality: 0.9, // Higher quality for better detection
      });
      
      if (!result.canceled && result.assets && result.assets.length > 0) {
        setOriginalImage(result.assets[0].uri);
        setProcessedImage(null);
        setDetectedObjects([]);
        setSelectedObject(null);
        console.log('Photo taken with camera');
        
        // Automatically detect objects after taking photo
        detectObjects(result.assets[0].uri);
      }
    } catch (err) {
      console.error('Camera error:', err);
      Alert.alert('Error', 'Could not open camera. Please try again.');
    }
  };

  const detectObjects = async (imageUri) => {
    try {
      setDetecting(true);
      console.log('Starting object detection...');
      
      const uriParts = imageUri.split('.');
      const fileType = uriParts[uriParts.length - 1];

      const formData = new FormData();
      formData.append('image', {
        uri: imageUri,
        name: `photo.${fileType}`,
        type: `image/${fileType}`,
      });

      console.log('Sending image for object detection...');
      
      const res = await fetch('http://192.168.114.130:5000/detect', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Detection response status:', res.status);

      if (!res.ok) {
        throw new Error(`Server responded with status: ${res.status}`);
      }

      const responseData = await res.json();
      console.log('Detection response received:', responseData);
      
      if (responseData.objects && responseData.objects.length > 0) {
        setDetectedObjects(responseData.objects);
        Alert.alert(
          "Objects Detected", 
          `Found ${responseData.objects.length} objects. Tap on any object to extract it.`
        );
      } else {
        Alert.alert("No Objects", "No objects were detected in the image.");
      }
      
    } catch (err) {
      console.error("Detection error:", err);
      Alert.alert("Error", `Failed to detect objects: ${err.message}`);
    } finally {
      setDetecting(false);
    }
  };

  const extractObject = async (objectData) => {
    try {
      setExtracting(true);
      setSelectedObject(objectData);
      console.log('Starting object extraction for:', objectData.class);
      
      const uriParts = originalImage.split('.');
      const fileType = uriParts[uriParts.length - 1];

      const formData = new FormData();
      formData.append('image', {
        uri: originalImage,
        name: `photo.${fileType}`,
        type: `image/${fileType}`,
      });
      
      // Send bounding box coordinates for precise extraction
      formData.append('bbox', JSON.stringify({
        x1: objectData.bbox[0],
        y1: objectData.bbox[1],
        x2: objectData.bbox[2],
        y2: objectData.bbox[3]
      }));
      
      formData.append('class', objectData.class);
      formData.append('confidence', objectData.confidence.toString());

      console.log('Sending for object extraction...');
      
      const res = await fetch('http://192.168.82.130:5000/extract', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Extraction response status:', res.status);

      if (!res.ok) {
        throw new Error(`Server responded with status: ${res.status}`);
      }

      const contentType = res.headers.get('content-type');
      console.log('Response content type:', contentType);

      if (contentType && contentType.includes('application/json')) {
        const responseData = await res.json();
        console.log('JSON response received');
        
        if (responseData.extractedImage) {
          setProcessedImage(`data:image/png;base64,${responseData.extractedImage}`);
        } else {
          throw new Error('Invalid JSON response format');
        }
      } else if (contentType && contentType.includes('image/')) {
        console.log('Binary image response received');
        const arrayBuffer = await res.arrayBuffer();
        const base64 = arrayBufferToBase64(arrayBuffer);
        const filename = FileSystem.documentDirectory + `extracted_${objectData.class}_${Date.now()}.png`;
        await FileSystem.writeAsStringAsync(filename, base64, {
          encoding: FileSystem.EncodingType.Base64,
        });
        setProcessedImage(filename);
      }

      Alert.alert("Success", `${objectData.class} extracted successfully!`);
      console.log('Object extraction completed');
      
    } catch (err) {
      console.error("Extraction error:", err);
      Alert.alert("Error", `Failed to extract object: ${err.message}`);
    } finally {
      setExtracting(false);
    }
  };

  const arrayBufferToBase64 = (buffer) => {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary);
  };

  const renderDetectedObjects = () => {
    if (detectedObjects.length === 0) return null;

    return (
      <View style={styles.objectsContainer}>
        <Text style={styles.objectsTitle}>Detected Objects (Tap to Extract):</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {detectedObjects.map((obj, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.objectCard,
                selectedObject?.class === obj.class && styles.selectedObjectCard
              ]}
              onPress={() => extractObject(obj)}
              disabled={extracting}
            >
              <Text style={styles.objectClass}>{obj.class}</Text>
              <Text style={styles.objectConfidence}>
                {(obj.confidence * 100).toFixed(1)}%
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>
    );
  };

  if (showAR && processedImage) {
    return (
      <View style={{ flex: 1 }}>
        <Camera style={{ flex: 1 }} type={Camera.Constants.Type.back} />
        <Image
          source={{ uri: processedImage }}
          style={styles.overlayImage}
          resizeMode="contain"
        />
      </View>
    );
  }

  return (
    <LinearGradient colors={["#6a11cb", "#2575fc"]} style={styles.background}>
      <SafeAreaView style={{ flex: 1 }}>
        <Animated.View style={{ flex: 1, opacity: fadeAnim }}>
          <ScrollView contentContainerStyle={styles.container}>
            <Text style={styles.title}>AR CutPaste Pro</Text>
            
            <View style={styles.debugBox}>
              <Text style={styles.debugText}>
                Camera: {String(cameraPermission)} | Media: {String(mediaPermission)}
              </Text>
            </View>

            {(cameraPermission !== 'granted' || mediaPermission !== 'granted') && (
              <View style={styles.permissionBox}>
                <Text style={styles.permissionText}>
                  {cameraPermission !== 'granted' && mediaPermission !== 'granted' 
                    ? 'Camera and Media Library permissions are required.'
                    : cameraPermission !== 'granted' 
                    ? 'Camera permission is required for taking photos.'
                    : 'Media Library permission is required for gallery access.'
                  }
                </Text>
                <Button title="Grant Permissions" onPress={requestPermissions} />
              </View>
            )}

            <View style={styles.buttonRow}>
              <View style={styles.buttonWrapper}>
                <Button 
                  title="Take Photo" 
                  onPress={takePhoto} 
                  disabled={cameraPermission !== 'granted' || detecting || extracting} 
                />
              </View>
              <View style={{ width: 16 }} />
              <View style={styles.buttonWrapper}>
                <Button 
                  title="Pick from Gallery" 
                  onPress={pickImage} 
                  disabled={mediaPermission !== 'granted' || detecting || extracting} 
                />
              </View>
            </View>

            {originalImage && (
              <View style={styles.imageBox}>
                <Text style={styles.label}>Original Image</Text>
                <Image source={{ uri: originalImage }} style={styles.image} />
              </View>
            )}

            {detecting && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#fff" />
                <Text style={styles.loadingText}>Detecting objects...</Text>
              </View>
            )}

            {renderDetectedObjects()}

            {extracting && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#fff" />
                <Text style={styles.loadingText}>
                  Extracting {selectedObject?.class}...
                </Text>
              </View>
            )}

            {processedImage && (
              <View style={styles.imageBox}>
                <Text style={styles.label}>
                  Extracted: {selectedObject?.class || 'Object'}
                </Text>
                <Image source={{ uri: processedImage }} style={styles.image} />
              </View>
            )}

            <View style={styles.arButtonContainer}>
              <Button 
                title="Place in AR" 
                onPress={() => setShowAR(true)} 
                disabled={!processedImage} 
              />
            </View>
          </ScrollView>
        </Animated.View>
      </SafeAreaView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
  },
  container: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 24,
    textShadowColor: '#000',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 4,
  },
  debugBox: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 8,
    padding: 8,
    marginBottom: 12,
    alignItems: 'center',
  },
  debugText: {
    color: '#fff',
    fontSize: 12,
    fontStyle: 'italic',
  },
  buttonRow: {
    flexDirection: 'row',
    marginBottom: 24,
  },
  buttonWrapper: {
    flex: 1,
  },
  permissionBox: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 8,
    padding: 16,
    marginBottom: 24,
    alignItems: 'center',
  },
  permissionText: {
    color: '#fff',
    fontSize: 14,
    marginBottom: 12,
    textAlign: 'center',
  },
  imageBox: {
    alignItems: 'center',
    marginVertical: 24,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  image: {
    width: 300,
    height: 300,
    borderRadius: 8,
  },
  loadingContainer: {
    alignItems: 'center',
    marginVertical: 24,
  },
  loadingText: {
    fontSize: 16,
    color: '#fff',
    marginTop: 8,
  },
  objectsContainer: {
    width: '100%',
    marginBottom: 24,
  },
  objectsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
    textAlign: 'center',
  },
  objectCard: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 8,
    padding: 12,
    marginRight: 12,
    alignItems: 'center',
  },
  selectedObjectCard: {
    backgroundColor: 'rgba(255,255,255,0.4)',
  },
  objectClass: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 4,
  },
  objectConfidence: {
    fontSize: 14,
    color: '#fff',
    fontStyle: 'italic',
  },
  arButtonContainer: {
    marginTop: 24,
    alignItems: 'center',
  },
  overlayImage: {
    position: 'absolute',
    top: Dimensions.get('window').height / 2 - 100, // Center vertically
    left: Dimensions.get('window').width / 2 - 100, // Center horizontally
    width: 200,
    height: 200,
    zIndex: 10,
  },
});
