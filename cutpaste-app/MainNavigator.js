import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import App from './App';
import ARScreen from './ARscreen';

const Stack = createStackNavigator();

export default function MainNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={App} />
        <Stack.Screen name="ARScreen" component={ARScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
