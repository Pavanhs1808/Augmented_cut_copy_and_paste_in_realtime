import React, { useState, useEffect, useRef } from 'react';
import {
  StatusBar,
  Platform,
  Pressable,
  Image,
  View,
  StyleSheet,
  ActivityIndicator,
  Alert,
  Text,
  SafeAreaView,
  ScrollView,
  Animated,
  TouchableOpacity,
  TextInput,
  Linking,
} from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { Camera, CameraView } from 'expo-camera';
import * as ExpoLinearGradient from 'expo-linear-gradient';

// --- NEW Pulsing Grid Loading Screen ---
const LoadingScreen = () => {
  const anims = useRef([...Array(9)].map(() => new Animated.Value(0))).current;

  useEffect(() => {
    const animations = anims.map((anim, i) => {
      return Animated.loop(
        Animated.sequence([
          Animated.timing(anim, {
            toValue: 1,
            duration: 600,
            delay: i * 200,
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0,
            duration: 600,
            useNativeDriver: true,
          }),
        ])
      );
    });
    Animated.parallel(animations).start();
  }, [anims]);

  return (
    <View style={styles.loadingContainer}>
      <StatusBar barStyle="light-content" />
      <View style={styles.loadingGrid}>
        {anims.map((anim, i) => (
          <Animated.View key={i} style={[styles.loadingDot, { opacity: anim }]} />
        ))}
      </View>
      <Text style={styles.loadingText}>Loading...</Text>
    </View>
  );
};


// LinearGradient fallback (keep existing behavior)
const LinearGradientComp = (ExpoLinearGradient?.default ?? ExpoLinearGradient?.LinearGradient) || null;
// Use CameraView for the component and Camera for the API
const CameraComp = CameraView;
const CameraAPI = Camera; // Camera also exposes permission helpers

const SERVER_URL_3D = 'http://10.198.127.130:5000'; // 3D Scan Server
const CLIPBOARD_SERVER_URL = 'http://10.198.127.130:8000'; // Clipboard Server

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
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

  // Use CameraAPI for constants (safer) and fallback strings
  const CAMERA_BACK = CameraAPI?.Constants?.Type?.back ?? 'back';
  const CAMERA_FRONT = CameraAPI?.Constants?.Type?.front ?? 'front';
  const [facing, setFacing] = useState(CAMERA_BACK);

  const [detectedText, setDetectedText] = useState('');
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Timer for the splash screen
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000); // 2 seconds

    // Animation for the main content
    Animated.timing(fadeAnim, { toValue: 1, duration: 600, useNativeDriver: true }).start();

    return () => clearTimeout(timer);
  }, [fadeAnim]);

  useEffect(() => {
    (async () => {
      try {
        const camPerm = await (CameraAPI?.getCameraPermissionsAsync?.() ?? CameraAPI?.requestCameraPermissionsAsync?.() ?? { status: null });
        const media = await ImagePicker.getMediaLibraryPermissionsAsync();
        setCameraPermission(camPerm?.status ?? null);
        setMediaPermission(media?.status ?? null);
      } catch {
        setCameraPermission(null);
        setMediaPermission(null);
      }
    })();
  }, []);

  const requestPermissions = async () => {
    try {
      const camRes = CameraAPI?.requestCameraPermissionsAsync
        ? await CameraAPI.requestCameraPermissionsAsync()
        : await ImagePicker.requestCameraPermissionsAsync?.();
      const mediaRes = await ImagePicker.requestMediaLibraryPermissionsAsync();

      const camStatus = camRes?.status ?? (camRes?.granted ? 'granted' : 'denied') ?? null;
      const mediaStatus = mediaRes?.status ?? (mediaRes?.granted ? 'granted' : 'denied') ?? null;

      setCameraPermission(camStatus);
      setMediaPermission(mediaStatus);

      if (camStatus !== 'granted' || mediaStatus !== 'granted') {
        Alert.alert(
          'Permissions',
          'Grant camera and media permissions in settings.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => Linking.openSettings() },
          ],
        );
        return false;
      }
      return true;
    } catch (e) {
      Alert.alert('Error', 'Failed to request permissions.');
      return false;
    }
  };

  const pickImage = async () => {
    if (mediaPermission !== 'granted') {
      Alert.alert('Permission', 'Media permission required', [{ text: 'Grant', onPress: requestPermissions }]);
      return;
    }
    try {
      const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, allowsEditing: false, quality: 0.9 });
      if (!result.canceled && result.assets?.length) {
        const uri = result.assets[0].uri;
        setOriginalImage(uri);
        setProcessedImage(null);
        setDetectedObjects([]);
        setSelectedObject(null);
        detectObjectsAndText(uri);
      }
    } catch {
      Alert.alert('Error', 'Could not open gallery.');
    }
  };

  const takePhoto = async () => {
    const ok = (cameraPermission === 'granted' && mediaPermission === 'granted')
      ? true
      : await requestPermissions();

    if (!ok) return;

    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        quality: 0.9,
      });
      if (!result.canceled && result.assets?.length) {
        const uri = result.assets[0].uri;
        setOriginalImage(uri);
        setProcessedImage(null);
        setDetectedObjects([]);
        setSelectedObject(null);
        detectObjectsAndText(uri);
      }
    } catch (err) {
      Alert.alert('Error', 'Could not open camera.');
    }
  };

  const detectObjectsAndText = async (imageUri) => {
    try {
      setDetecting(true);
      const ext = (imageUri.split('.').pop() || 'jpg').replace(/\?.*$/, '');
      const formData = new FormData();
      formData.append('image', { uri: imageUri, name: `photo.${ext}`, type: `image/${ext}` });
      const res = await fetch(`${SERVER_URL_3D}/detect_and_ocr`, { method: 'POST', body: formData, headers: { 'Content-Type': 'multipart/form-data' } });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = await res.json();
      setDetectedObjects(data.objects || []);
      setDetectedText(data.text || '');
      if (data.objects?.length) Alert.alert('Objects found', `${data.objects.length} objects detected — tap to extract.`);
      else Alert.alert('No objects', 'No objects detected.');
    } catch (err) {
      Alert.alert('Error', `Detection failed: ${err.message || err}`);
    } finally {
      setDetecting(false);
    }
  };

  const extractObject = async (obj) => {
    try {
      setExtracting(true);
      setSelectedObject(obj);
      const ext = (originalImage?.split('.').pop() || 'jpg').replace(/\?.*$/, '');
      const form = new FormData();
      form.append('image', { uri: originalImage, name: `photo.${ext}`, type: `image/${ext}` });
      form.append('bbox', JSON.stringify({ x1: obj.bbox[0], y1: obj.bbox[1], x2: obj.bbox[2], y2: obj.bbox[3] }));
      form.append('class', obj.class);
      form.append('confidence', String(obj.confidence));
      const res = await fetch(`${SERVER_URL_3D}/extract`, { method: 'POST', body: form, headers: { 'Content-Type': 'multipart/form-data' } });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        const j = await res.json();
        if (j.extractedImage) setProcessedImage(`data:image/png;base64,${j.extractedImage}`);
      } else if (ct.includes('image/')) {
        const buf = await res.arrayBuffer();
        const b64 = arrayBufferToBase64(buf);
        const path = FileSystem.documentDirectory + `extracted_${obj.class}_${Date.now()}.png`;
        await FileSystem.writeAsStringAsync(path, b64, { encoding: FileSystem.EncodingType.Base64 });
        setProcessedImage(path);
      }
      Alert.alert('Done', `${obj.class} extracted`);
    } catch (err) {
      Alert.alert('Error', `Extract failed: ${err.message || err}`);
    } finally {
      setExtracting(false);
    }
  };

  const handleUpload = async () => {
    try {
      const result = await ImagePicker.launchCameraAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Videos, allowsEditing: false, quality: 1 });
      if (!result.canceled && result.assets?.length) {
        const videoUri = result.assets[0].uri;
        const ext = (videoUri.split('.').pop() || 'mp4').replace(/\?.*$/, '');
        const form = new FormData();
        form.append('video', { uri: videoUri, name: `video.${ext}`, type: `video/${ext}` });
        setUploading(true);
        const res = await fetch(`${SERVER_URL_3D}/extract_frames`, { method: 'POST', body: form, headers: { 'Content-Type': 'multipart/form-data' } });
        const json = await res.json().catch(() => ({}));
        setUploading(false);
        if (res.ok) Alert.alert('Uploaded', 'Video uploaded. Processing started on server.');
        else Alert.alert('Server error', json.message || 'Upload failed');
      }
    } catch (err) {
      setUploading(false);
      Alert.alert('Error', `Upload failed: ${err.message || err}`);
    }
  };

  const sendTextToPCClipboard = async (text) => {
    try {
      const res = await fetch(`${CLIPBOARD_SERVER_URL}/paste_text`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
      if (res.ok) Alert.alert('Sent', 'Text sent to PC clipboard');
      else Alert.alert('Error', 'Failed to send text to PC');
    } catch (err) {
      Alert.alert('Error', `${err.message || err}`);
    }
  };

  const sendImageToPC = async () => {
    if (!processedImage) {
      return Alert.alert('No Image', "An object hasn't been extracted yet.");
    }
    try {
      let base64Img = processedImage.startsWith('data:')
        ? processedImage.split(',')[1]
        : await FileSystem.readAsStringAsync(processedImage, { encoding: FileSystem.EncodingType.Base64 });

      const res = await fetch(`${CLIPBOARD_SERVER_URL}/paste`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Img }),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }
      
      Alert.alert('Success', 'Image sent to PC clipboard.');
    } catch (err) {
      Alert.alert('Error', `Failed to send image: ${err.message || err}`);
    }
  };

  const arrayBufferToBase64 = (buffer) => {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      for (let j = 0; j < chunk.length; j++) binary += String.fromCharCode(chunk[j]);
    }
    if (typeof btoa === 'function') return btoa(binary);
    if (typeof Buffer !== 'undefined') return Buffer.from(binary, 'binary').toString('base64');
    throw new Error('No base64 encoder');
  };

  const headerTop = Platform.OS === 'android' ? (StatusBar.currentHeight || 24) + 8 : 18;

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (showAR) {
    const Cam = CameraComp;
    return (
      <View style={{ flex: 1 }}>
        <Cam style={{ flex: 1 }} facing={facing}>
          <View style={styles.arOverlay}>
            <TouchableOpacity style={styles.backButton} onPress={() => setShowAR(false)}><Text style={styles.backButtonText}>← Back</Text></TouchableOpacity>
            {processedImage ? (
              <View pointerEvents="box-none" style={styles.centerOverlayWrapper}>
                <Image source={{ uri: processedImage }} style={styles.centerOverlay} resizeMode="contain" />
              </View>
            ) : null}
            <TouchableOpacity style={styles.flipButton} onPress={() => setFacing(prev => (prev === CAMERA_BACK ? CAMERA_FRONT : CAMERA_BACK))}><Text style={styles.flipButtonText}>Flip</Text></TouchableOpacity>
            
            {processedImage && (
              <TouchableOpacity style={styles.sendToPCButton} onPress={sendImageToPC}>
                <Text style={styles.sendToPCText}>Send to PC</Text>
              </TouchableOpacity>
            )}
          </View>
        </Cam>
      </View>
    );
  }

  const Gradient = LinearGradientComp || (({ children, style }) => <View style={style}>{children}</View>);

  return (
    <Gradient colors={['#F3F4F8', '#E6E9F2']} style={styles.safe}>
      <SafeAreaView style={{ flex: 1 }}>
        <StatusBar barStyle="dark-content" backgroundColor="#F3F4F8" />
        <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
          <Animated.View style={[styles.card, { opacity: fadeAnim }]}>
            <Text style={styles.title}>Object & Text Scanner</Text>

            <View style={styles.previewBox}>
              {!originalImage && !processedImage && (
                <View style={styles.placeholder}>
                  <Text style={styles.placeholderTitle}>Drop image here</Text>
                  <Text style={styles.placeholderSubtitle}>or pick one to get started</Text>
                </View>
              )}
              {originalImage && !processedImage && <Image source={{ uri: originalImage }} style={styles.previewImage} resizeMode="contain" />}
              {processedImage && <Image source={{ uri: processedImage }} style={styles.previewImage} resizeMode="contain" />}
              {detecting && <View style={styles.overlayLoading}><ActivityIndicator size="large" color="#6A5ACD" /><Text style={styles.loadingTextNew}>Analyzing...</Text></View>}
            </View>

            <View style={styles.row}>
              <Pressable style={[styles.actionBtn, styles.primaryBtn]} onPress={pickImage}><Text style={styles.actionText}>Pick Image</Text></Pressable>
              <Pressable style={[styles.actionBtn, styles.secondaryBtn]} onPress={takePhoto}><Text style={styles.actionText}>Take Photo</Text></Pressable>
              <Pressable style={[styles.actionBtn, {backgroundColor: '#A094E1'}]} onPress={handleUpload} disabled={uploading}><Text style={styles.actionText}>{uploading ? 'Uploading...' : '3D Scan'}</Text></Pressable>
            </View>

            {detectedObjects.length > 0 && (
              <View style={styles.objectsContainer}>
                <Text style={styles.objectsTitle}>Detected Objects</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ paddingVertical: 8 }}>
                  {detectedObjects.map((obj, i) => (
                    <TouchableOpacity key={i} style={[styles.objectCard, selectedObject?.class === obj.class && styles.selectedObjectCard]} onPress={() => extractObject(obj)} disabled={extracting}>
                      <Text style={styles.objectClass}>{obj.class}</Text>
                      <Text style={styles.objectConfidence}>{Math.round((obj.confidence || 0) * 100)}%</Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>
            )}

            <View style={styles.textCard}>
              <Text style={styles.textTitle}>Detected Text</Text>
              <TextInput
                value={detectedText}
                onChangeText={setDetectedText}
                editable
                multiline
                placeholder="Detected text will appear here. You can edit, copy or clear."
                style={styles.textInput}
              />
              <View style={styles.textActions}>
                <Pressable style={styles.smallBtn} onPress={() => { if (!detectedText) return Alert.alert('No text'); import('expo-clipboard').then(c => c.setStringAsync(detectedText)); Alert.alert('Copied', 'Text copied to device clipboard'); }}>
                  <Text style={styles.smallBtnText}>Copy</Text>
                </Pressable>
                <Pressable style={[styles.smallBtn, { backgroundColor: '#ef4444' }]} onPress={() => { setDetectedText(''); }}>
                  <Text style={styles.smallBtnText}>Clear</Text>
                </Pressable>
                <Pressable style={[styles.smallBtn, { backgroundColor: '#6A5ACD' }]} onPress={() => sendTextToPCClipboard(detectedText)}>
                  <Text style={styles.smallBtnText}>Send to PC</Text>
                </Pressable>
              </View>
            </View>

            <View style={styles.center}>
              <Pressable
                style={styles.bigBtn}
                onPress={async () => {
                  const ok = (cameraPermission === 'granted' && mediaPermission === 'granted')
                    ? true
                    : await requestPermissions();
                  if (ok) setShowAR(true);
                }}
              >
                <Text style={styles.bigBtnText}>Open AR View</Text>
              </Pressable>
            </View>

          </Animated.View>
        </ScrollView>
      </SafeAreaView>
    </Gradient>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#111827'
  },
  loadingGrid: {
    width: 150,
    height: 150,
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  loadingDot: {
    width: '33.33%',
    height: '33.33%',
    padding: 10, // Creates space between dots
    transform: [{ scale: 0.5 }], // Make dots smaller
    backgroundColor: '#81E6D9',
    borderRadius: 50, // Make them circles
  },
  loadingText: {
    marginTop: 40, // Increased margin
    fontSize: 18,
    fontWeight: 'bold',
    color: '#E5E7EB',
  },
  safe: { flex: 1 },
  container: { padding: 24, alignItems: 'center' },
  card: { 
    width: '100%', 
    maxWidth: 760, 
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    borderRadius: 20,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 10,
  },
  title: { 
    fontSize: 28, 
    fontWeight: 'bold', 
    marginBottom: 20, 
    color: '#333'
  },
  previewBox: { 
    width: '100%', 
    height: 360, 
    borderRadius: 16, 
    backgroundColor: '#FFF', 
    overflow: 'hidden', 
    justifyContent: 'center', 
    alignItems: 'center', 
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#EEE'
  },
  placeholder: { alignItems: 'center' },
  placeholderTitle: { fontSize: 18, fontWeight: '600', color: '#555' },
  placeholderSubtitle: { fontSize: 14, color: '#AAA', marginTop: 8 },
  previewImage: { width: '100%', height: '100%' },
  overlayLoading: { 
    position: 'absolute', 
    alignItems: 'center', 
    justifyContent: 'center', 
    backgroundColor: 'rgba(255,255,255,0.9)', 
    width: '100%', 
    height: '100%' 
  },
  loadingTextNew: { marginTop: 10, color: '#6A5ACD', fontWeight: '600' },
  row: { 
    width: '100%', 
    flexDirection: 'row', 
    justifyContent: 'space-around',
    gap: 12,
    marginBottom: 16 
  },
  actionBtn: { 
    flex: 1, 
    paddingVertical: 16, 
    borderRadius: 12, 
    alignItems: 'center' 
  },
  primaryBtn: { backgroundColor: '#6A5ACD' },
  secondaryBtn: { backgroundColor: '#8A7DDE' },
  actionText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
  objectsContainer: { width: '100%', marginTop: 12, marginBottom: 12 },
  objectsTitle: { fontSize: 18, fontWeight: 'bold', color: '#444', marginBottom: 8 },
  objectCard: { 
    backgroundColor: '#fff',
    padding: 14, 
    marginRight: 10, 
    borderRadius: 12, 
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#EAEAEA'
  },
  selectedObjectCard: { borderColor: '#6A5ACD', borderWidth: 2 },
  objectClass: { fontWeight: 'bold', color: '#333' },
  objectConfidence: { color: '#777', marginTop: 4 },

  textCard: { width: '100%', backgroundColor: '#fff', padding: 16, borderRadius: 12, borderWidth: 1, borderColor: '#EAEAEA', marginTop: 10 },
  textTitle: { fontWeight: 'bold', fontSize: 16, color: '#444', marginBottom: 10 },
  textInput: { 
    minHeight: 100, 
    maxHeight: 180, 
    backgroundColor: '#F9F9F9', 
    padding: 12, 
    borderRadius: 10, 
    borderWidth: 1, 
    borderColor: '#EAEAEA',
    color: '#333'
  },
  textActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: 10, marginTop: 10 },
  smallBtn: { backgroundColor: '#8A7DDE', paddingHorizontal: 14, paddingVertical: 10, borderRadius: 8 },
  smallBtnText: { color: '#fff', fontWeight: 'bold' },

  center: { width: '100%', alignItems: 'center', marginTop: 20 },
  bigBtn: {
    backgroundColor: '#6A5ACD',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 30,
    minWidth: 240,
    alignItems: 'center',
    shadowColor: "#6A5ACD",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 8
  },
  bigBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 20 },

  backButton: { position: 'absolute', top: Platform.OS === 'android' ? 34 : 54, left: 16, zIndex: 10 },
  backButtonText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  flipButton: { position: 'absolute', top: Platform.OS === 'android' ? 34 : 54, right: 16, zIndex: 10, backgroundColor: 'rgba(0,0,0,0.45)', paddingHorizontal: 10, paddingVertical: 8, borderRadius: 8 },
  flipButtonText: { color: '#fff', fontWeight: '700' },

  arOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  centerOverlayWrapper: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 220,
    height: 220,
    marginLeft: -110,
    marginTop: -110,
    alignItems: 'center',
    justifyContent: 'center',
  },
  centerOverlay: {
    width: 200,
    height: 200,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#fff',
    backgroundColor: 'transparent',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.5,
    shadowRadius: 4,
  },

  sendToPCButton: {
    position: 'absolute',
    bottom: 40,
    backgroundColor: '#6A5ACD',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
    alignSelf: 'center',
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  sendToPCText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
