import React, { useEffect } from 'react';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import './Login.css';
import img from './Stroke.png';

function Login() {
  const [username, setUsername] = React.useState(null);
  const [password, setPassword] = React.useState(null);
  return (
    <div className="container">
      <div className="Login">
        <TextField
          label="Username"
          id="standard-basic"
          onChange={(e) => setUsername(String(e.target.value))}
          value={username}
        />
        <TextField
          label="Password"
          id="standard-basic"
          onChange={(e) => setPassword(String(e.target.value))}
          value={password}
        />
      </div>
    </div>
  );
}

export default Login;
