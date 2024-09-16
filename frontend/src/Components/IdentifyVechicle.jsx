import React, { useEffect, useRef, useState } from 'react'
import '../Components/monitorVechicle.css'
import Sidebars from './sidebar'
import { Outlet } from 'react-router-dom'

function IdentifyVechicle() {

    const [file, setFile] = useState();
    const [data, setData] = useState();
    const [intervalId, setIntervalId] = useState(null);
    const fileInputRef = useRef(null);

    const [message, setMessage] = useState([]);

    useEffect(() => {
        if (file) {
            sendFileToBackend(file)
            console.log('File data:', file);
        }
    }, [file]);

    function handleChange(e) {
        console.log(e.target.files);
        const fileData = new Blob(e.target.files);
        setFile(fileData);
        setData(URL.createObjectURL(e.target.files[0]));
        // setFile(URL.createObjectURL(e.target.files[0]));
    }

    async function sendFileToBackend(file) {
        try {
            const formData = new FormData();
            formData.append('file', file, 'image');

            const response = await fetch('http://127.0.0.1:8000/detect-number-plate/', {
                method: 'POST',
                body: formData
            });
            response.json().then(data => {
                console.log(data.messages);
                setMessage(data.messages)// Access the JSON data here
            }).catch(error => {
                console.error('Error fetching JSON:', error);
            });

            if (!response.ok) {
                throw new Error('Failed to upload file');
            }
            setMessage('File uploaded successfully');

            console.log('File uploaded successfully');
        } catch (error) {
            setMessage('Error uploading file');
            console.error('Error uploading file:', error);
        }
    }


    const handleStart = async () => {
        clearFileInput();
        try {
            const initialResponse = await fetch('http://127.0.0.1:8000/start', { method: 'POST' });
            if (!initialResponse.ok) {
                alert('Failed to start detection');
                return; // Exit if initial call fails

            }
            const initialData = await initialResponse.json(); // Assuming the response is JSON
            console.log("start", initialData);
            // window.location.reload(); // Reload the page to clear file input and image display


        } catch (error) {
            console.error('Error starting detection:', error);
            setMessage("failed to start");
            alert('Failed to start detection');
            return; // Exit if initial call fails
        }

        const intervalId = setInterval(async () => {
            try {
                const repeatedResponse = await fetch('http://127.0.0.1:8000/latest-messages', { method: 'GET' }); // Replace with your actual endpoint
                if (!repeatedResponse.ok) {
                    console.error('Error during repeated call:', repeatedResponse.statusText);
                    return; // Handle errors gracefully
                }
                const repeatedData = await repeatedResponse.json(); // Assuming the response is JSON
                console.log("repeated call", repeatedData);
                setMessage(repeatedData); // Update UI or state with the received data
            } catch (error) {
                console.error('Error during repeated call:', error);
            }
        }, 15000); // Set your desired interval (15 seconds in this example)

        // Store the interval ID for potential clearing later (optional)
        setIntervalId(intervalId);
    };

    const handleStop = () => {
        clearInterval(intervalId);
        fetch('http://127.0.0.1:8000/stop', { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    console.log("stop", response);
                } else {
                    alert('Failed to stop detection');
                }
            })
            .catch(error => {
                console.error('Error stopping detection:', error);
                alert('Failed to stop detection');
            });
    };

    function clearFileInput() {
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
            setFile(null);
            setData('');
        }
    }


    return (
        <div >
            <h2>Monitor Vechicle</h2>
            <div className='mt-4 d-flex flex-column justify-content-center align-items-center'>
                {data && <img className='preview-image' src={data} alt="Preview" />}
                <input type="file" ref={fileInputRef} onChange={handleChange} />
            </div>
            <div className='mt-5'>
                <h3>Live Stream</h3>
                <button type='button' className="btn me-md-2 btn-outline-primary" onClick={handleStart}>start</button>
                <button type='button' className="btn md-2 btn-outline-danger" onClick={handleStop}>stop</button>
            </div>
            <div className='d-flex justify-content-center mt-4'>
                <div className='Register-vehicle bg-light p-4 message-box' >
                    <h3>Message Box</h3>
                    {message.map((message, index) => (
                        <div key={index} className={message.includes('successfully') ? 'success-message' : 'error-message'}>
                            {message}
                        </div>
                    ))}                </div>
            </div>

        </div>
    );
}

export default IdentifyVechicle