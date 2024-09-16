import React, { useState } from 'react';
import '../Components/Register.css';
import 'bootstrap/dist/css/bootstrap.min.css';

function RegisterVechicle() {
    const [vehicleNo, setVehicleNo] = useState('');
    const [ownerName, setOwnerName] = useState('');
    const [phoneNo, setPhoneNo] = useState('');
    // const [file, setFile] = useState(null);
    const [imageUploaded, setImageUploaded] = useState(false);
    const[imageData,setImageData]=useState();

    const handleFileChange = (e) => {
        // setFile(e.target.files[0]);
        const imageFile =new Blob(e.target.files);
        setImageData(imageFile);
        setImageUploaded(true);
        // Clear vehicle number when uploading image
        setVehicleNo('');
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        try {

            if (imageUploaded) {
                // formData.append('image', imageData);
                // formData = { ...formData, 'image': imageData };
                const formData = new FormData();
                formData.append('owner_name', ownerName);
                formData.append('phone_number', phoneNo);
                formData.append('image',imageData);

                const imageUploadResponse = await fetch('http://127.0.0.1:8000/register-vehicle-from-image/', {
                    method: 'POST',
                    body: formData
                });

                if (imageUploadResponse.ok) {
                    const imageData = await imageUploadResponse.json();
                    console.log(imageData);
                    setVehicleNo('');
                    setOwnerName('');
                    setPhoneNo('');
                    setImageUploaded(false);
                    console.log('Image uploaded successfully.');
                } else {
                    throw new Error('Failed to upload image.');
                }
            } else if (vehicleNo) {
                let formData = {
                  owner_name:ownerName,
                  phone_number:phoneNo
                };
                // formData.append('owner_name', ownerName);
                // formData.append('vehicle_number', vehicleNo);
                // formData.append('phone_number', phoneNo);
                formData = { ...formData, 'vehicle_number': vehicleNo };

                const response = await fetch('http://127.0.0.1:8000/register-vehicle/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                if (response.ok) {
                    const responseData = await response.json();
                    console.log(responseData);
                    setVehicleNo('');
                    setOwnerName('');
                    setPhoneNo('');
                    console.log('Form data submitted successfully.');
                } else {
                    throw new Error('Failed to submit form data.');
                }
            } else {
                throw new Error('Please enter either vehicle number or upload an image.');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };



    
    const handleClear = () => {
        setVehicleNo('');
        setOwnerName('');
        setPhoneNo('');
        setImageUploaded(false);
    };

    return (
        <div id='container-fluid'>
            <h2 className='d-flex justify-content-center align-items-center mt-4' style={{ width: '100%' }}>
                Register Vehicle Page
            </h2>
            <div className="d-flex justify-content-center align-items-center mt-4">
                <div className="box-container">
                    <div className='Register-vehicle bg-light p-4'>
                        <h3 className='text-secondary font-styles'>Register Manually</h3>
                        <div className="form-group">
                            <label className='basic-font-styles'>Enter Owner Name</label>
                            <input
                                type='text'
                                className="form-control"
                                placeholder="Enter owner name"
                                value={ownerName}
                                onChange={(e) => setOwnerName(e.target.value)}
                            />
                        </div>
                        <div className="form-group">
                            <label className='basic-font-styles'>Enter Phone No</label>
                            <input
                                type='text'
                                className="form-control"
                                placeholder="Enter phone no"
                                value={phoneNo}
                                onChange={(e) => setPhoneNo(e.target.value)}
                                
                            />
                        </div>
                    </div>
                </div>
            </div>

            <h3>Register Vehicle Number</h3>
            <div style={{ width: '100%' }} className='down-page '>
                <div className="box-container ">
                    <div className="form-group Register-vehicle bg-light p-4">
                        <label className='basic-font-styles'>Enter Vehicle No</label>
                        <input
                            type='text'
                            className="form-control"
                            placeholder="Enter vehicle no"
                            value={vehicleNo}
                            onChange={(e) => setVehicleNo(e.target.value)}
                            disabled={imageUploaded} // Disable vehicle number field if image is uploaded
                        />
                    </div>
                    <div className='mt-4 mb-1'>
                        <hr className='hr-tag ' />
                        <span className="text-center text-secondary font-styles">or</span>
                    </div>
                    <h4 className='mt-4'>Register using upload file</h4>
                    <div className='box-container-new Register-vehicle bg-light p-4'>
                        <div className='d-flex '>
                            <input type='file' onChange={handleFileChange} disabled={!!vehicleNo} />
                            {/* Disable upload button if vehicle number is entered */}
                            <button className="btn btn-primary btn-sm " disabled={!!vehicleNo}>Upload</button>
                        </div>
                    </div>
                </div>
            </div>
            <div className='d-flex justify-content-center mt-5 '>
                <button className="btn btn-primary btn-sm me-2" onClick={handleSubmit}>Submit</button>
                <button className="btn btn-secondary btn-sm" onClick={handleClear}>Clear</button>
            </div>
        </div>
    );
}

export default RegisterVechicle;
