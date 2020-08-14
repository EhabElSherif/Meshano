import { Ionicons } from '@expo/vector-icons';
import * as React from 'react';

import Colors from '../constants/Colors';

export default function TabBarIcon(props) {
  return (
    <Ionicons
      name={props.name}
      size={30}
      style={[styles.icon]}
      color={props.focused ? Colors.tabIconSelected : Colors.tabIconDefault}
    />
  );
}
const styles = {
  icon:{
    marginBottom: -7
  }
}