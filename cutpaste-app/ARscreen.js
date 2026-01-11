import React from 'react';
import { ViroARScene, ViroText, ViroARSceneNavigator } from '@reactvision/react-viro';

function HelloAR() {
  return (
    <ViroARScene>
      <ViroText text="Hello AR!" position={[0, 0, -1]} />
    </ViroARScene>
  );
}

export default function ARScreen() {
  return <ViroARSceneNavigator initialScene={{ scene: HelloAR }} />;
}